from tqdm import tqdm
import argparse
from utils import *
import random
from client import *
from freebase_func import SPARQLPATH, set_sparql_path
from probe_utils import (
    candidate_rows,
    gold_answers_for,
    print_question_log,
    question_id_for,
    retained_rows,
    summarize_question,
    write_probe_outputs,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--sparql_endpoint", type=str,
                        default="", help="Freebase SPARQL endpoint. Overrides TOG_SPARQL_ENDPOINT.")
    parser.add_argument("--probe_output", type=str,
                        default="", help="If set, write minimal evidence probe outputs to this directory.")
    parser.add_argument("--probe_limit", type=int,
                        default=0, help="If >0, run only this many questions for the probe.")
    args = parser.parse_args()

    if args.sparql_endpoint:
        set_sparql_path(args.sparql_endpoint)
    if "xxx.xxx.xxx.xxx" in SPARQLPATH and not args.sparql_endpoint:
        raise SystemExit(
            "Missing Freebase endpoint. Set --sparql_endpoint http://HOST:PORT/sparql "
            "or TOG_SPARQL_ENDPOINT before running ToG."
        )

    datas, question_string = prepare_dataset(args.dataset)
    if args.probe_limit > 0:
        datas = datas[:args.probe_limit]

    probe_rows = []
    total_questions = len(datas)
    for probe_index, data in enumerate(tqdm(datas), start=1):
        question = data[question_string]
        topic_entity = data['topic_entity']
        cluster_chain_of_entities = []
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        probe_row = None
        if args.probe_output:
            probe_row = {
                "question_id": question_id_for(data, probe_index),
                "question": question,
                "gold_answers": gold_answers_for(data, args.dataset),
                "tog_final_answer": "",
                "tog_correct": False,
                "candidate_entities_generated": [],
                "retained_beam_entities": [],
                "path_trace": [],
            }
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                if entity['head']:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                else:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                
                if len(entity_candidates_id) >=20:
                    entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                if len(entity_candidates_id) ==0:
                    continue

                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
            
            if len(total_candidates) ==0:
                answer = half_stop(question, cluster_chain_of_entities, args)
                if probe_row is not None:
                    probe_row["tog_final_answer"] = answer
                break

            depth_candidates = candidate_rows(
                depth,
                total_entities_id,
                total_relations,
                total_candidates,
                total_topic_entities,
                total_head,
                total_scores,
            )
            if probe_row is not None:
                probe_row["candidate_entities_generated"].extend(depth_candidates)
                probe_row["path_trace"].extend([row["path_trace"] for row in depth_candidates])
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            if probe_row is not None:
                probe_row["retained_beam_entities"].extend(retained_rows(depth_candidates, args.width))
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    if probe_row is not None:
                        probe_row["tog_final_answer"] = results
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                    continue
            else:
                answer = half_stop(question, cluster_chain_of_entities, args)
                if probe_row is not None:
                    probe_row["tog_final_answer"] = answer
                flag_printed = True
                break
        
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            if probe_row is not None:
                probe_row["tog_final_answer"] = results

        if probe_row is not None:
            summarize_question(probe_row)
            print_question_log(probe_row, probe_index, total_questions)
            probe_rows.append(probe_row)

    if args.probe_output:
        write_probe_outputs(args.probe_output, probe_rows)
