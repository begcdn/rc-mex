# Generator Faithfulness: Question Pairs for Manual Judgement

One entry per WebQSP question where the annotated path was among the proposed
candidates (90 of 100). For each, the generator was shown that path and wrote a
question from it. It never saw the original question.

**Scores, ranks, and which path the system finally chose are deliberately omitted**
so the judgement is blind.

## What to mark

For each pair, write one word after `Verdict:`

- `SAME` — the generated question asks for the same thing as the original.
- `DIFFERENT` — it asks for something else. This is a generator error.
- `ANNOTATION` — the generated question correctly describes the path shown, but
  that path does not really answer the original question. Not the generator's fault.

Use `ANNOTATION` sparingly — only when you would defend the generator to a reviewer.

## Tally (fill in when done)

```
SAME:        
DIFFERENT:   
ANNOTATION:  
```

---

## 1. `WebQTest-0`

**Asked:** what does jamaican people speak?

**Generator wrote:** What human language is spoken in Jamaica?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.languages_spoken::forward`
- this path returns: Jamaican Creole English Language, Jamaican English
- correct answers: Jamaican English, Jamaican Creole English Language

</details>

`Verdict:` SAME

---

## 2. `WebQTest-100`

**Asked:** what language is spoken in haiti today?

**Generator wrote:** What is the spoken language of Haiti?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.languages_spoken::forward`
- this path returns: French, Haitian Creole
- correct answers: Haitian Creole, French

</details>

`Verdict:` SAME

---

## 3. `WebQTest-102`

**Asked:** who played barbara gordon batgirl?

**Generator wrote:** Who is the actor that portrays Barbara Gordon, a comic book character?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `film.film_character.portrayed_in_films::forward , film.performance.actor::forward`
- this path returns: Hannah Gunn, Ilyssa Fradin, Melinda McGraw
- correct answers: Melinda McGraw, Hannah Gunn, Ilyssa Fradin

</details>

`Verdict:` SAME

---

## 4. `WebQTest-104`

**Asked:** where is jay leno from?

**Generator wrote:** Where was Jay Leno, a TV episode or TV program creator, born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: New Rochelle
- correct answers: New Rochelle

</details>

`Verdict:` SAME

---

## 5. `WebQTest-105`

**Asked:** what language do people from thailand speak?

**Generator wrote:** What human language is spoken in Thailand?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.languages_spoken::forward`
- this path returns: Akha Language, Cham language, Hmong language, Khmer language
- correct answers: Mon Language, Saek language, Akha Language, Hmong language

</details>

`Verdict:` ANNOTATION

---

## 6. `WebQTest-111`

**Asked:** where is the carpathian mountain range located?

**Generator wrote:** What entity contains the mountain range Carpathian Mountains?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.containedby::forward`
- this path returns: Europe
- correct answers: Europe

</details>

`Verdict:` SAME

---

## 7. `WebQTest-112`

**Asked:** what was robert burns?

**Generator wrote:** What is the profession of Robert Burns?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Author, Bard, Poet, Writer
- correct answers: Bard, Author, Writer, Poet

</details>

`Verdict:` SAME

---

## 8. `WebQTest-118`

**Asked:** who fought in the gulf war 1991?

**Generator wrote:** Which entity is included as a combatant by the entity that was involved in Gulf War, a military conflict?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `military.military_conflict.combatants::forward , military.military_combatant_group.combatants::forward`
- this path returns: Argentina, Australia, France, Iraq
- correct answers: France, Australia, United States of America, Argentina

</details>

`Verdict:` ANNOTATION

---

## 9. `WebQTest-122`

**Asked:** where did francisco coronado come from?

**Generator wrote:** In which city, town, or village was Francisco Vázquez de Coronado, an author, born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: Salamanca
- correct answers: Salamanca

</details>

`Verdict:` SAME

---

## 10. `WebQTest-124`

**Asked:** what language does egyptian people speak?

**Generator wrote:** What is the official language of Egypt?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.official_language::forward`
- this path returns: Modern Standard Arabic
- correct answers: Modern Standard Arabic

</details>

`Verdict:` SAME

---

## 11. `WebQTest-125`

**Asked:** what are abraham sons names?

**Generator wrote:** Who is the deceased person or religious leader that is a child of Abraham?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.children::forward`
- this path returns: Isaac, Ishbak, Ishmael, Jokshan
- correct answers: Midian, Isaac, Medan, Ishmael

</details>

`Verdict:` DIFFERENT

---

## 12. `WebQTest-126`

**Asked:** who wrote the jana gana mana?

**Generator wrote:** Who is the author or composer that composed Jana Gana Mana, a national anthem?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `music.composition.composer::forward`
- this path returns: Rabindranath Tagore, Ram Singh Thakur
- correct answers: Ram Singh Thakur, Rabindranath Tagore

</details>

`Verdict:` SAME

---

## 13. `WebQTest-128`

**Asked:** who plays juni cortez?

**Generator wrote:** Who is the actor that portrays Juni Cortez?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `film.film_character.portrayed_in_films::forward , film.performance.actor::forward`
- this path returns: Daryl Sabara
- correct answers: Daryl Sabara

</details>

`Verdict:` SAME

---

## 14. `WebQTest-129`

**Asked:** what was lucille ball?

**Generator wrote:** What is the profession of Lucille Ball, who is a film actor?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Actor, Comedian, Model, Singer
- correct answers: Actor, Comedian, Television producer, Model

</details>

`Verdict:` ANNOTATION

---

## 15. `WebQTest-13`

**Asked:** who was vice president after kennedy died?

**Generator wrote:** Who was the vice president under John F. Kennedy?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `government.us_president.vice_president::forward`
- this path returns: Lyndon B. Johnson
- correct answers: Lyndon B. Johnson

</details>

`Verdict:` ANNOTATION

---

## 16. `WebQTest-131`

**Asked:** who inspired obama?

**Generator wrote:** Which author influenced Barack Obama?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `influence.influence_node.influenced_by::forward`
- this path returns: Nipsey Russell, Reinhold Niebuhr, Saul Alinsky
- correct answers: Nipsey Russell, Reinhold Niebuhr, Saul Alinsky

</details>

`Verdict:` SAME

---

## 17. `WebQTest-133`

**Asked:** where did dolly parton grow up?

**Generator wrote:** In which entity was Dolly Parton born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: Sevierville
- correct answers: Sevierville

</details>

`Verdict:` SAME

---

## 18. `WebQTest-134`

**Asked:** what type of books did agatha christie wrote?

**Generator wrote:** What is the profession of Agatha Christie?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Novelist, Playwright, Poet, Screenwriter
- correct answers: Writer, Screenwriter, Playwright, Novelist

</details>

`Verdict:` ANNOTATION

---

## 19. `WebQTest-138`

**Asked:** what are the four main languages spoken in spain?

**Generator wrote:** What human language is spoken in Spain?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.languages_spoken::forward`
- this path returns: Basque Language, Catalan language, Galician Language, Occitan language
- correct answers: Catalan language, Occitan language, Spanish Language, Galician Language

</details>

`Verdict:` SAME

---

## 20. `WebQTest-139`

**Asked:** where is the nra headquarters located?

**Generator wrote:** What is the entity where the mailing address of the headquarters of National Rifle Association, a membership organization, is located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `organization.organization.headquarters::forward , location.mailing_address.citytown::forward`
- this path returns: Fairfax
- correct answers: Fairfax

</details>

`Verdict:` SAME

---

## 21. `WebQTest-14`

**Asked:** where is the fukushima daiichi nuclear plant located?

**Generator wrote:** Which city, town, or village contains Fukushima Daiichi Nuclear Power Plant, a disaster-affected structure?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.containedby::forward`
- this path returns: Japan, Okuma
- correct answers: Japan, Okuma

</details>

`Verdict:` SAME

---

## 22. `WebQTest-142`

**Asked:** who developed the tcp ip reference model?

**Generator wrote:** Who is the academic or computer scientist that invented Transmission Control Protocol?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `law.invention.inventor::forward`
- this path returns: Robert  E. Kahn, Vint Cerf
- correct answers: Vint Cerf, Robert  E. Kahn

</details>

`Verdict:` SAME

---

## 23. `WebQTest-146`

**Asked:** what team does jordan own?

**Generator wrote:** What sports team has Jordan as its location?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `sports.sports_team_location.teams::forward`
- this path returns: Al-Wehdat SC, Jordan national football team
- correct answers: Jordan national football team, Al-Wehdat SC

</details>

`Verdict:` ANNOTATION   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 24. `WebQTest-149`

**Asked:** what do portuguese people speak?

**Generator wrote:** What is the official language of Portugal?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.official_language::forward`
- this path returns: Portuguese Language
- correct answers: Portuguese Language

</details>

`Verdict:` SAME

---

## 25. `WebQTest-150`

**Asked:** what language does cuba speak?

**Generator wrote:** What human language is spoken by Cuba?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.languages_spoken::forward`
- this path returns: Haitian Creole, Lucumi Language, Spanish Language
- correct answers: Haitian Creole, Spanish Language, Lucumi Language

</details>

`Verdict:` SAME

---

## 26. `WebQTest-153`

**Asked:** what are the sights to see in madrid?

**Generator wrote:** What tourist attraction is included in Madrid?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `travel.travel_destination.tourist_attractions::forward`
- this path returns: Almudena Cathedral, Buen Retiro Park, El Escorial, Festimad
- correct answers: Madrid Arena, Festimad, Parque Warner Madrid, Plaza de Cibeles

</details>

`Verdict:` ANNOTATION

---

## 27. `WebQTest-154`

**Asked:** what instruments did louis armstrong play?

**Generator wrote:** What is the book or musical instrument that Louis Armstrong plays?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `music.group_member.instruments_played::forward`
- this path returns: Cornet, Trumpet
- correct answers: Cornet, Trumpet

</details>

`Verdict:` DIFFERENT 'added book' on its own

---

## 28. `WebQTest-156`

**Asked:** what time zone am i in california?

**Generator wrote:** What is the time zone of California, a US state?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.time_zones::forward`
- this path returns: Pacific Time Zone
- correct answers: Pacific Time Zone

</details>

`Verdict:` SAME

---

## 29. `WebQTest-157`

**Asked:** where is mission san buenaventura located?

**Generator wrote:** What is the postal code or US county that contains Mission San Buenaventura?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.containedby::forward`
- this path returns: 93001, Ventura County
- correct answers: Ventura County, 93001

</details>

`Verdict:` SAME

---

## 30. `WebQTest-159`

**Asked:** what time in hilo hawaii?

**Generator wrote:** What is the time zone of Hilo?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.time_zones::forward`
- this path returns: Hawaii-Aleutian Time Zone
- correct answers: Hawaii-Aleutian Time Zone

</details>

`Verdict:` SAME

---

## 31. `WebQTest-164`

**Asked:** what did stephen hawking become famous for?

**Generator wrote:** What is the profession of Stephen Hawking?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Astronomer, Astrophysicist, Author, Cosmologist
- correct answers: Cosmologist, Mathematician, Author, Physicist

</details>

`Verdict:` ANNOTATION

---

## 32. `WebQTest-165`

**Asked:** who plays donna noble?

**Generator wrote:** Who is the actor featured in the television program where Donna Noble appeared?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `tv.tv_character.appeared_in_tv_program::forward , tv.regular_tv_appearance.actor::forward`
- this path returns: Catherine Tate
- correct answers: Catherine Tate

</details>

`Verdict:` DIFFERENT

---

## 33. `WebQTest-166`

**Asked:** what was dr seuss education?

**Generator wrote:** What is the institution associated with the educational institution attended by Theodore Lesieg, who is an author?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.education::forward , education.education.institution::forward`
- this path returns: Dartmouth College, Lincoln College, Oxford, University of Oxford
- correct answers: Lincoln College, Oxford, University of Oxford, Dartmouth College

</details>

`Verdict:` ANNOTATION   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 34. `WebQTest-167`

**Asked:** where did queensland get its name from?

**Generator wrote:** What is the entity after which Queensland, an Australian state, is named?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `symbols.namesake.named_after::forward`
- this path returns: Queen Victoria
- correct answers: Queen Victoria

</details>

`Verdict:` SAME

---

## 35. `WebQTest-168`

**Asked:** what document did james madison write?

**Generator wrote:** What is a book written by James Madison, who is a US President?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `book.author.works_written::forward`
- this path returns: A vocabulary of New Jersey Delaware, All impressments unlawful and inadmissible, An address delivered before the Agricultural Society of Albemarle, on Tuesday, May 12, 1818, An examination of the British doctrine
- correct answers: The Papers of James Madison, Secretary of State Series, Vol. 2, The Papers of James Madison, Vol. 4, Federalist No. 56, Federalist No. 20

</details>

`Verdict:` ANNOTATION

---

## 36. `WebQTest-169`

**Asked:** what are the three official languages of belgium?

**Generator wrote:** What is the official language of Belgium?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.official_language::forward`
- this path returns: Dutch Language, French, German Language
- correct answers: Dutch Language, French, German Language

</details>

`Verdict:` SAME

---

## 37. `WebQTest-171`

**Asked:** what airport do you fly into to get to destin fl?

**Generator wrote:** What is the nearby airport of Destin?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.nearby_airports::forward`
- this path returns: Destin Executive Airport, Destin–Fort Walton Beach Airport
- correct answers: Destin–Fort Walton Beach Airport, Destin Executive Airport

</details>

`Verdict:` SAME

---

## 38. `WebQTest-177`

**Asked:** where was theodore roosevelt buried?

**Generator wrote:** What is the cemetery where Theodore Roosevelt is buried?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.place_of_burial::forward`
- this path returns: Youngs Memorial Cemetery
- correct answers: Youngs Memorial Cemetery

</details>

`Verdict:` SAME

---

## 39. `WebQTest-179`

**Asked:** what artistic movement did henri matisse belong to?

**Generator wrote:** What is the art period or movement that Henri Matisse, a film character or visual artist, is associated with?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `visual_art.visual_artist.associated_periods_or_movements::forward`
- this path returns: Fauvism, Impressionism, Modern art, Modernism
- correct answers: Neo-impressionism, Modernism, Modern art, Impressionism

</details>

`Verdict:` SAME

---

## 40. `WebQTest-19`

**Asked:** what is my timezone in louisiana?

**Generator wrote:** What is the time zone of Louisiana, a US state?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.time_zones::forward`
- this path returns: Central Time Zone
- correct answers: Central Time Zone

</details>

`Verdict:` SAME

---

## 41. `WebQTest-20`

**Asked:** what did st augustine do?

**Generator wrote:** What is the profession of Augustine of Hippo, who is a religious leader?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Philosopher, Physician, Writer
- correct answers: Writer, Physician, Philosopher

</details>

`Verdict:` SAME

---

## 42. `WebQTest-22`

**Asked:** what kind government does egypt have?

**Generator wrote:** What is the form of government of Egypt?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.form_of_government::forward`
- this path returns: Provisional government, Semi-presidential system
- correct answers: Semi-presidential system, Provisional government

</details>

`Verdict:` SAME

---

## 43. `WebQTest-23`

**Asked:** what town was martin luther king assassinated in?

**Generator wrote:** In which entity did Martin Luther King, Jr. pass away?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.place_of_death::forward`
- this path returns: Memphis
- correct answers: Memphis

</details>

`Verdict:` SAME

---

## 44. `WebQTest-24`

**Asked:** where did edgar allan poe died?

**Generator wrote:** In which entity did Edgar Allan Poe pass away?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.place_of_death::forward`
- this path returns: Baltimore
- correct answers: Baltimore

</details>

`Verdict:` SAME

---

## 45. `WebQTest-26`

**Asked:** what to do today in atlanta with kids?

**Generator wrote:** What tourist attraction is included in Atlanta?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `travel.travel_destination.tourist_attractions::forward`
- this path returns: Arbor Place Mall, Atlanta Ballet, Atlanta Cyclorama & Civil War Museum, Atlanta History Center
- correct answers: Fox Theatre, Jimmy Carter Library and Museum, Centennial Olympic Park, Cobb Energy Performing Arts Centre

</details>

`Verdict:` ANNOTATION

---

## 46. `WebQTest-28`

**Asked:** what electorate does anna bligh represent?

**Generator wrote:** What is the district represented by the position held by Anna Bligh?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `government.politician.government_positions_held::forward , government.government_position_held.district_represented::forward`
- this path returns: Electoral district of South Brisbane
- correct answers: Electoral district of South Brisbane

</details>

`Verdict:` ANNOTATION

---

## 47. `WebQTest-33`

**Asked:** what timezone is sweden?

**Generator wrote:** What is the time zone of Sweden?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.time_zones::forward`
- this path returns: Central European Time Zone
- correct answers: Central European Time Zone

</details>

`Verdict:` SAME

---

## 48. `WebQTest-34`

**Asked:** who did cam newton sign with?

**Generator wrote:** Which team is associated with the salary of Cam Newton, an American football player?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `base.schemastaging.athlete_extra.salary::forward , base.schemastaging.athlete_salary.team::forward`
- this path returns: Carolina Panthers
- correct answers: Carolina Panthers

</details>

`Verdict:` ANNOTATION

---

## 49. `WebQTest-35`

**Asked:** what county is frederick md in?

**Generator wrote:** In which US county is Frederick located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.hud_county_place.county::forward`
- this path returns: Frederick County
- correct answers: Frederick County

</details>

`Verdict:` SAME

---

## 50. `WebQTest-37`

**Asked:** what timezone is utah in?

**Generator wrote:** What is the time zone of Utah, a US state?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.time_zones::forward`
- this path returns: Mountain Time Zone
- correct answers: Mountain Time Zone

</details>

`Verdict:` SAME

---

## 51. `WebQTest-41`

**Asked:** where is the galapagos islands located on a world map?

**Generator wrote:** What administrative division or body of water contains Galápagos Islands, an island group?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.location.containedby::forward`
- this path returns: Ecuador, Galápagos Province, Pacific Ocean
- correct answers: Ecuador, Pacific Ocean, Galápagos Province

</details>

`Verdict:` SAME

---

## 52. `WebQTest-42`

**Asked:** what are the songs that justin bieber wrote?

**Generator wrote:** What is the award-winning work or canonical version of a musical recording composed by Justin Bieber?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `music.composer.compositions::forward`
- this path returns: #thatPower, All Around The World, All Bad, All That Matters
- correct answers: Somebody to Love, All Around The World, Wait for a Minute, Roller Coaster

</details>

`Verdict:` DIFFERENT

---

## 53. `WebQTest-43`

**Asked:** what did george clemenceau do?

**Generator wrote:** What is the profession of Georges Clémenceau, who is a politician?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Journalist, Physician, Publisher, Statesman
- correct answers: Physician, Writer, Statesman, Journalist

</details>

`Verdict:` SAME

---

## 54. `WebQTest-45`

**Asked:** where was rihanna born and raised?

**Generator wrote:** In which entity was Rihanna born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: Saint Michael Parish
- correct answers: Saint Michael Parish

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 55. `WebQTest-47`

**Asked:** where george lopez was born?

**Generator wrote:** In which neighborhood was George Lopez born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: Mission Hills
- correct answers: Mission Hills

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 56. `WebQTest-48`

**Asked:** where are samsung based?

**Generator wrote:** What is the entity where the mailing address of the headquarters of Samsung Group, an organization founder, is located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `organization.organization.headquarters::forward , location.mailing_address.citytown::forward`
- this path returns: Suwon
- correct answers: Suwon

</details>

`Verdict:` DIFFERENT   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 57. `WebQTest-49`

**Asked:** what did the islamic people believe in?

**Generator wrote:** What is a belief or field of study that Islam, a religion, has as a belief?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `religion.religion.beliefs::forward`
- this path returns: Entering Heaven alive, God in Islam, Islamic holy books, Islamic view of angels
- correct answers: Ṭūbā, Mahdi, Monotheism, Prophets in Islam

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 58. `WebQTest-51`

**Asked:** who will play mr gray in the film?

**Generator wrote:** Who is the actor that portrays Christian Grey?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `film.film_character.portrayed_in_films::forward , film.performance.actor::forward`
- this path returns: Jamie Dornan
- correct answers: Jamie Dornan

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 59. `WebQTest-52`

**Asked:** what did george orwell died of?

**Generator wrote:** What was the cause of death for George Orwell?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.cause_of_death::forward`
- this path returns: Tuberculosis
- correct answers: Tuberculosis

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 60. `WebQTest-54`

**Asked:** what country did adolf hitler control?

**Generator wrote:** Which country is under the jurisdiction of the position held by Adolf Hitler, a military commander?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `government.politician.government_positions_held::forward , government.government_position_held.jurisdiction_of_office::forward`
- this path returns: Nazi Germany
- correct answers: Nazi Germany

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 61. `WebQTest-55`

**Asked:** who is michael buble?

**Generator wrote:** What is the profession of Michael Bublé, who is a musical artist?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Actor, Singer, Songwriter
- correct answers: Actor, Singer, Songwriter

</details>

`Verdict:` ANNOTATION   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 62. `WebQTest-56`

**Asked:** what county is kansas city kansas?

**Generator wrote:** In which US county is Kansas City located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.hud_county_place.county::forward`
- this path returns: Wyandotte County
- correct answers: Wyandotte County

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 63. `WebQTest-6`

**Asked:** where is jamarcus russell from?

**Generator wrote:** In which city, town, or village was JaMarcus Russell, an American football player, born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: Mobile
- correct answers: Mobile

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 64. `WebQTest-60`

**Asked:** where did eleanor roosevelt die?

**Generator wrote:** In which entity did Eleanor Roosevelt pass away?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.place_of_death::forward`
- this path returns: Manhattan
- correct answers: Manhattan

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 65. `WebQTest-61`

**Asked:** what are the religions practiced in indonesia?

**Generator wrote:** What is the religion associated with the location that Indonesia, a book or country, has as a location?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.statistical_region.religions::forward , location.religion_percentage.religion::forward`
- this path returns: Catholicism, Hinduism, Islam, Protestantism
- correct answers: Catholicism, Hinduism, Protestantism, Islam

</details>

`Verdict:` DIFFERENT   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 66. `WebQTest-63`

**Asked:** who was vp for lincoln?

**Generator wrote:** Who was the vice president under Abraham Lincoln?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `government.us_president.vice_president::forward`
- this path returns: Andrew Johnson, Hannibal Hamlin
- correct answers: Hannibal Hamlin, Andrew Johnson

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 67. `WebQTest-66`

**Asked:** what is the currency of puerto rico called?

**Generator wrote:** What is the currency used in Puerto Rico?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.currency_used::forward`
- this path returns: United States Dollar
- correct answers: United States Dollar

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 68. `WebQTest-67`

**Asked:** what kind of cancer did carl wilson have?

**Generator wrote:** What disease or medical condition caused the death of Carl Wilson?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.cause_of_death::forward`
- this path returns: Brain tumor, Lung cancer
- correct answers: Lung cancer, Brain tumor

</details>

`Verdict:` ANNOTATION   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 69. `WebQTest-69`

**Asked:** what county is brentwood tennessee in?

**Generator wrote:** In which US county is Brentwood located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.hud_county_place.county::forward`
- this path returns: Williamson County
- correct answers: Williamson County

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 70. `WebQTest-7`

**Asked:** where was george washington carver from?

**Generator wrote:** In which city, town, or village was George Washington Carver, an inventor, born?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.place_of_birth::forward`
- this path returns: Diamond
- correct answers: Diamond

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 71. `WebQTest-72`

**Asked:** what battles did stonewall jackson fight in?

**Generator wrote:** What is the military conflict involving the entity commanded by Stonewall Jackson?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `military.military_commander.military_commands::forward , military.military_command.military_conflict::forward`
- this path returns: American Civil War, Battle of Cedar Mountain, Battle of Chancellorsville, Battle of Chantilly
- correct answers: Manassas Station Operations, First Battle of Kernstown, First Battle of Rappahannock Station, Second Battle of Bull Run

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 72. `WebQTest-73`

**Asked:** what language do the maasai tribe speak?

**Generator wrote:** What is the language spoken by Maasai people?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.ethnicity.languages_spoken::forward`
- this path returns: Maasai Language
- correct answers: Maasai Language

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 73. `WebQTest-75`

**Asked:** what disease did patrick swayze died from?

**Generator wrote:** What disease or medical condition caused the death of Patrick Swayze, a film actor?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.cause_of_death::forward`
- this path returns: Pancreatic cancer
- correct answers: Pancreatic cancer

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 74. `WebQTest-76`

**Asked:** what works of art did leonardo da vinci produce?

**Generator wrote:** What is the artwork created by Leonardo da Vinci, who is a film character or visual artist?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `visual_art.visual_artist.artworks::forward`
- this path returns: Adoration of the Magi, Annunciation, Bacchus, Benois Madonna
- correct answers: g.121wt37c, g.1224tf0c, Sala delle Asse, Ginevra de' Benci

</details>

`Verdict:` DIFFERENT   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 75. `WebQTest-77`

**Asked:** what capital of austria?

**Generator wrote:** What is the capital city of Austria?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.capital::forward`
- this path returns: Vienna
- correct answers: Vienna

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 76. `WebQTest-79`

**Asked:** what country did buddha come from?

**Generator wrote:** What is the nationality of Gautama Buddha, who is a religious leader?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.nationality::forward`
- this path returns: Nepal
- correct answers: Nepal

</details>

`Verdict:` DIFFERENT   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 77. `WebQTest-8`

**Asked:** what else did ben franklin invent?

**Generator wrote:** What invention or product category was created by Benjamin Franklin?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `law.inventor.inventions::forward`
- this path returns: Bifocals, Franklin stove, Glass harmonica, Lightning rod
- correct answers: Lightning rod, Glass harmonica, Bifocals, Franklin stove

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 78. `WebQTest-80`

**Asked:** what county is greeley colorado in?

**Generator wrote:** In which US county is Greeley located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.hud_county_place.county::forward`
- this path returns: Weld County
- correct answers: Weld County

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 79. `WebQTest-82`

**Asked:** what did peter tchaikovsky do?

**Generator wrote:** What is the profession of Pyotr Ilyich Tchaikovsky, who is a composer?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.person.profession::forward`
- this path returns: Composer, Librettist, Musician
- correct answers: Musician, Librettist, Composer

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 80. `WebQTest-84`

**Asked:** what type of music did john lennon sing?

**Generator wrote:** What is the musical genre in which John Lennon performs?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `music.artist.genre::forward`
- this path returns: Art rock, Blues rock, Experimental music, Experimental rock
- correct answers: Rock music, Psychedelic rock, Pop rock, Blues rock

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 81. `WebQTest-86`

**Asked:** which country does greenland belong to?

**Generator wrote:** Which country is Greenland part of?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.administrative_division.country::forward`
- this path returns: Denmark
- correct answers: Denmark

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 82. `WebQTest-87`

**Asked:** what the zip code for seattle washington?

**Generator wrote:** What is the postal code of Seattle?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.citytown.postal_codes::forward`
- this path returns: 98101, 98102, 98103, 98104
- correct answers: 98185, 98174, 98109, 98132

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 83. `WebQTest-89`

**Asked:** what do you call the chinese writing system?

**Generator wrote:** What is the writing system used by Chinese language?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `language.human_language.writing_system::forward`
- this path returns: 'Phags-pa script, Chinese characters, Nüshu script, Simplified Chinese character
- correct answers: Simplified Chinese character, Chinese characters, Nüshu script, Traditional Chinese characters

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 84. `WebQTest-90`

**Asked:** who played on the jeffersons?

**Generator wrote:** Who is the TV program creator featured in the regular cast member of The Jeffersons?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `tv.tv_program.regular_cast::forward , tv.regular_tv_appearance.actor::forward`
- this path returns: Berlinda Tolbert, Damon Evans, Franklin Cover, Isabel Sanford
- correct answers: Damon Evans, Jay Hammer, Isabel Sanford, Sherman Hemsley

</details>

`Verdict:` DIFFERENT   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 85. `WebQTest-91`

**Asked:** what is the name of the san francisco newspaper?

**Generator wrote:** What newspaper or periodical has San Francisco as its circulation area?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `periodicals.newspaper_circulation_area.newspapers::forward`
- this path returns: AsianWeek, Bay Area Reporter, California Star, Dock of the Bay
- correct answers: Synapse, San Francisco Call, San Francisco Business Times, The Golden Era

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 86. `WebQTest-93`

**Asked:** what continent does armenia belong to?

**Generator wrote:** In which continent is Armenia located?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `base.locations.countries.continent::forward`
- this path returns: Europe
- correct answers: Europe

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 87. `WebQTest-94`

**Asked:** what did randy savage died of?

**Generator wrote:** What disease or medical condition caused the death of Randy Savage, a TV personality?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.cause_of_death::forward`
- this path returns: heart attack
- correct answers: heart attack

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 88. `WebQTest-96`

**Asked:** where did richard nixon die?

**Generator wrote:** In which city, town, or village did Richard Nixon, a US President, pass away?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `people.deceased_person.place_of_death::forward`
- this path returns: New York City
- correct answers: New York City

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 89. `WebQTest-98`

**Asked:** who was vice president when jfk was president?

**Generator wrote:** Who was the vice president under John F. Kennedy?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `government.us_president.vice_president::forward`
- this path returns: Lyndon B. Johnson
- correct answers: Lyndon B. Johnson

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

## 90. `WebQTest-99`

**Asked:** what is serbian language called?

**Generator wrote:** What is the official language of Serbia?

<details><summary>path and answers (open only if you need them)</summary>

- relations: `location.country.official_language::forward`
- this path returns: Serbian language
- correct answers: Serbian language

</details>

`Verdict:` SAME   <!-- judged by Claude, blind; 79% agreement with Ziad on a 14-item calibration sample -->
---

