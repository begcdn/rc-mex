# Are These Paths Right, or Just Lucky?

Each block is a question, its correct answer, and questions generated from paths
that **return exactly that answer** but are not the path WebQSP annotated.

For each, mark whether it is a genuine way of asking the original question, or
whether it only lands on the right answer by coincidence.

- `y` — a valid alternative reading. Different words, same request.
- `n` — reaches the answer through unrelated reasoning. Coincidence.
- `?` — the original question is too vague to tell.

## Why this matters

Systems like RoG are trained on paths that reach the gold answer. Only 29% of
questions have a unique such path; the rest admit several. If most alternatives
are `y`, that redundancy is harmless. If most are `n`, then the supervision every
such system trains on is mostly noise — and that is a claim about the field, not
about our pipeline.

Our own comparator corpus counts all of these as positives, so this also measures
how much noise we trained on.

---

## 1. `WebQTest-0`

**what does jamaican people speak?**

correct answer: Jamaican English, Jamaican Creole English Language

- [y] What human language is spoken in Jamaica?  ``  <!-- labelled by Claude -->

---

## 2. `WebQTest-6`

**where is jamarcus russell from?**

correct answer: Mobile

- [y] What is the city, town, or village where JaMarcus Russell, an American football player, has lived?  ``  <!-- labelled by Claude -->

- [n] What is the city or town that is located in the same county as the city where JaMarcus Russell, an American football player, was born?  ``  <!-- labelled by Claude -->

- [n] In which city, town, or village has an entity lived who has also lived with JaMarcus Russell, an American football player?  ``  <!-- labelled by Claude -->

- [n] Which city, town, or village is located in the same county as the city where JaMarcus Russell, an American football player, was born?  ``  <!-- labelled by Claude -->

---

## 3. `WebQTest-7`

**where was george washington carver from?**

correct answer: Diamond

- [n] Which city, town, or village is located near the city, town, or village where George Washington Carver, an inventor, was born?  ``  <!-- labelled by Claude -->

- [n] What is the city, town, or village that is located in the same county as the city, town, or village where George Washington Carver, an inventor, was born?  ``  <!-- labelled by Claude -->

- [y] In which city, town, or village was George Washington Carver, an inventor, born?  ``  <!-- labelled by Claude -->

- [n] Which city, town, or village is located in the same city, town, or village as the birthplace of George Washington Carver, an inventor?  ``  <!-- labelled by Claude -->

---

## 4. `WebQTest-8`

**what else did ben franklin invent?**

correct answer: Lightning rod, Glass harmonica, Bifocals, Franklin stove

- [n] What invention was created by the film character that Benjamin Franklin is based on?  ``  <!-- labelled by Claude -->

- [n] What invention was created by the film character based on Benjamin Franklin?  ``  <!-- labelled by Claude -->

- [y] What invention was created by Benjamin Franklin, a film character?  ``  <!-- labelled by Claude -->

---

## 5. `WebQTest-13`

**who was vice president after kennedy died?**

correct answer: Lyndon B. Johnson

- [n] Which US President is John F. Kennedy associated with?  ``  <!-- labelled by Claude -->

---

## 6. `WebQTest-19`

**what is my timezone in louisiana?**

correct answer: Central Time Zone

- [n] What is the time zone of the capital city of Louisiana, a US state?  ``  <!-- labelled by Claude -->

---

## 7. `WebQTest-22`

**what kind government does egypt have?**

correct answer: Semi-presidential system, Provisional government

- [y] What is the form of government in Egypt?  ``  <!-- labelled by Claude -->

---

## 8. `WebQTest-23`

**what town was martin luther king assassinated in?**

correct answer: Memphis

- [n] Where did the artwork depicted in Martin Luther King, Jr. pass away?  ``  <!-- labelled by Claude -->

- [n] Where did the artwork representing Martin Luther King, Jr. pass away?  ``  <!-- labelled by Claude -->

- [y] What is the entity that Martin Luther King, Jr., an artwork, passed away in?  ``  <!-- labelled by Claude -->

- [n] What is the entity that is located at the same place as the entity where Martin Luther King, Jr., an artwork, passed away?  ``  <!-- labelled by Claude -->

---

## 9. `WebQTest-24`

**where did edgar allan poe died?**

correct answer: Baltimore

- [y] What is the entity where the author Edgar Allan Poe passed away?  ``  <!-- labelled by Claude -->

- [n] Where did the author who was translated from Edgar Allan Poe pass away?  ``  <!-- labelled by Claude -->

- [n] Where did the author represented in fiction by Edgar Allan Poe pass away?  ``  <!-- labelled by Claude -->

- [n] Where did the author who is an edition of Edgar Allan Poe pass away?  ``  <!-- labelled by Claude -->

---

## 10. `WebQTest-26`

**what to do today in atlanta with kids?**

correct answer: Fox Theatre, Jimmy Carter Library and Museum, Centennial Olympic Park, Cobb Energy Performing Arts Centre

- [?] What venue is near Atlanta?  ``  <!-- labelled by Claude -->

---

## 11. `WebQTest-28`

**what electorate does anna bligh represent?**

correct answer: Electoral district of South Brisbane

- [y] What is the district represented by the position held by Anna Bligh, a politician?  ``  <!-- labelled by Claude -->

- [y] What entity is represented by the position held by Anna Bligh, a politician?  ``  <!-- labelled by Claude -->

- [y] What entity is represented by the position held by Anna Bligh, who is a politician?  ``  <!-- labelled by Claude -->

---

## 12. `WebQTest-33`

**what timezone is sweden?**

correct answer: Central European Time Zone

- [n] What is the time zone of the capital city of Sweden?  ``  <!-- labelled by Claude -->

- [n] What is the time zone of the location containing Sweden?  ``  <!-- labelled by Claude -->

---

## 13. `WebQTest-34`

**who did cam newton sign with?**

correct answer: Carolina Panthers

- [n] Which team is associated with the salary of Cam Newton, an American football player?  ``  <!-- labelled by Claude -->

---

## 14. `WebQTest-35`

**what county is frederick md in?**

correct answer: Frederick County

- [n] What is the US county whose county seat is a city or town that is located in Frederick?  ``  <!-- labelled by Claude -->

- [y] In which US county is Frederick located?  ``  <!-- labelled by Claude -->

- [n] What is the US county where a city or town that is located in Frederick is situated?  ``  <!-- labelled by Claude -->

- [n] Which US county has Frederick as its county seat?  ``  <!-- labelled by Claude -->

---

## 15. `WebQTest-37`

**what timezone is utah in?**

correct answer: Mountain Time Zone

- [n] What is the time zone of the capital city of Utah, a US state?  ``  <!-- labelled by Claude -->

---

## 16. `WebQTest-42`

**what are the songs that justin bieber wrote?**

correct answer: Somebody to Love, All Around The World, Wait for a Minute, Roller Coaster

- [y] What is the composition that Justin Bieber composed?  ``  <!-- labelled by Claude -->

---

## 17. `WebQTest-45`

**where was rihanna born and raised?**

correct answer: Saint Michael Parish

- [n] What entity is part of the country that Rihanna, a book, originates from?  ``  <!-- labelled by Claude -->

- [n] What is the place where the book written by Rihanna was born?  ``  <!-- labelled by Claude -->

- [n] What is a first-level administrative division of the country that Rihanna, a book, originates from?  ``  <!-- labelled by Claude -->

- [n] Where was the author of Rihanna born?  ``  <!-- labelled by Claude -->

---

## 18. `WebQTest-47`

**where george lopez was born?**

correct answer: Mission Hills

- [n] What is the neighborhood where the musical recording that performed or created George Lopez was born?  ``  <!-- labelled by Claude -->

- [n] In which neighborhood was the season included in George Lopez born?  ``  <!-- labelled by Claude -->

- [n] What is the neighborhood that is located in the same area as the neighborhood where George Lopez, a musical recording, was born?  ``  <!-- labelled by Claude -->

- [n] In which neighborhood was the track included in George Lopez born?  ``  <!-- labelled by Claude -->

---

## 19. `WebQTest-49`

**what did the islamic people believe in?**

correct answer: Ṭūbā, Mahdi, Monotheism, Prophets in Islam

- [y] What is a belief of Islam, a religion?  ``  <!-- labelled by Claude -->

---

## 20. `WebQTest-51`

**who will play mr gray in the film?**

correct answer: Jamie Dornan

- [n] What is an entity that shares the same gender as Christian Grey, a film character?  ``  <!-- labelled by Claude -->

- [y] Who is the actor for the performance that portrays Christian Grey, a film character?  ``  <!-- labelled by Claude -->

- [n] Who acted in the film in which Christian Grey, a film character, was portrayed?  ``  <!-- labelled by Claude -->

- [y] Who acted in the performance that portrays Christian Grey, a film character?  ``  <!-- labelled by Claude -->

---

## 21. `WebQTest-52`

**what did george orwell died of?**

correct answer: Tuberculosis

- [n] What was the cause of death for the fictional character based on George Orwell?  ``  <!-- labelled by Claude -->

- [n] What was the cause of death for the author who is represented in fiction as George Orwell?  ``  <!-- labelled by Claude -->

- [n] What was the cause of death of the author who George Orwell is based on?  ``  <!-- labelled by Claude -->

---

## 22. `WebQTest-54`

**what country did adolf hitler control?**

correct answer: Nazi Germany

- [n] Which country has the military commander that was commanded by Adolf Hitler, a military commander?  ``  <!-- labelled by Claude -->

---

## 23. `WebQTest-56`

**what county is kansas city kansas?**

correct answer: Wyandotte County

- [n] Which US county has Kansas City as its county seat?  ``  <!-- labelled by Claude -->

- [y] In which US county is Kansas City located?  ``  <!-- labelled by Claude -->

- [n] What is the US county where a city or town that is located in Kansas City is situated?  ``  <!-- labelled by Claude -->

- [n] Which US county is a second-level division of the entity where Kansas City is located?  ``  <!-- labelled by Claude -->

---

## 24. `WebQTest-60`

**where did eleanor roosevelt die?**

correct answer: Manhattan

- [n] What is the county seat of the entity where Eleanor Roosevelt passed away?  ``  <!-- labelled by Claude -->

- [n] Where did the book that has Eleanor Roosevelt as its subject pass away?  ``  <!-- labelled by Claude -->

- [n] Where did the book that represents Eleanor Roosevelt pass away?  ``  <!-- labelled by Claude -->

- [n] What entity is administratively part of the city, town, or village where Eleanor Roosevelt, a book, was born?  ``  <!-- labelled by Claude -->

---

## 25. `WebQTest-66`

**what is the currency of puerto rico called?**

correct answer: United States Dollar

- [y] What is the currency used in Puerto Rico?  ``  <!-- labelled by Claude -->

---

## 26. `WebQTest-67`

**what kind of cancer did carl wilson have?**

correct answer: Lung cancer, Brain tumor

- [n] What is the cause of death for the album released by Carl Wilson, a composer?  ``  <!-- labelled by Claude -->

- [n] What is the cause of death for the composer who released an album titled Carl Wilson?  ``  <!-- labelled by Claude -->

- [n] What was the cause of death for the composer who performs the album Carl Wilson?  ``  <!-- labelled by Claude -->

- [n] What disease or medical condition caused the death of the composer who is the primary release of Carl Wilson?  ``  <!-- labelled by Claude -->

---

## 27. `WebQTest-69`

**what county is brentwood tennessee in?**

correct answer: Williamson County

- [y] In which US county is Brentwood located?  ``  <!-- labelled by Claude -->

- [n] In which US county is a city or town that is located in Brentwood located?  ``  <!-- labelled by Claude -->

- [n] What is the US county where a city or town that is located in Brentwood is situated?  ``  <!-- labelled by Claude -->

---

## 28. `WebQTest-72`

**what battles did stonewall jackson fight in?**

correct answer: Manassas Station Operations, First Battle of Kernstown, First Battle of Rappahannock Station, Second Battle of Bull Run

- [n] Which military conflict involved the commander commanded by Stonewall Jackson, a composer?  ``  <!-- labelled by Claude -->

---

## 29. `WebQTest-75`

**what disease did patrick swayze died from?**

correct answer: Pancreatic cancer

- [y] What disease or medical condition is Patrick Swayze, a film actor, associated with?  ``  <!-- labelled by Claude -->

- [n] What disease or medical condition is the parent classification of the medical condition associated with Patrick Swayze, a film actor?  ``  <!-- labelled by Claude -->

- [y] What is the disease or medical condition that Patrick Swayze, a film actor, has?  ``  <!-- labelled by Claude -->

- [n] What disease or medical condition is included in the same ICD-9 CM classification as the cause of death of Patrick Swayze, a film actor?  ``  <!-- labelled by Claude -->

---

## 30. `WebQTest-77`

**what capital of austria?**

correct answer: Vienna

- [y] What is the capital of Austria?  ``  <!-- labelled by Claude -->

---

## 31. `WebQTest-79`

**what country did buddha come from?**

correct answer: Nepal

- [y] What is the entity containing the city, town, or village where Gautama Buddha, a religious leader, was born?  ``  <!-- labelled by Claude -->

- [n] What is the nationality of the deceased person who is a child of Gautama Buddha, a religious leader?  ``  <!-- labelled by Claude -->

---

## 32. `WebQTest-80`

**what county is greeley colorado in?**

correct answer: Weld County

- [n] What is the US county where a city or town that is located in Greeley is situated?  ``  <!-- labelled by Claude -->

- [n] What is the US county whose county seat is a city or town that is located in Greeley?  ``  <!-- labelled by Claude -->

- [n] Which US county has Greeley as its county seat?  ``  <!-- labelled by Claude -->

- [y] In which US county is Greeley located?  ``  <!-- labelled by Claude -->

---

## 33. `WebQTest-86`

**which country does greenland belong to?**

correct answer: Denmark

- [n] What is the main country for the language spoken in Greenland?  ``  <!-- labelled by Claude -->

- [n] Which country has a spoken language that is also the main language of Greenland?  ``  <!-- labelled by Claude -->

- [n] Which country speaks the language that has Greenland as its main country?  ``  <!-- labelled by Claude -->

---

## 34. `WebQTest-89`

**what do you call the chinese writing system?**

correct answer: Simplified Chinese character, Chinese characters, Nüshu script, Traditional Chinese characters

- [y] What is the language writing system used by Chinese language?  ``  <!-- labelled by Claude -->

---

## 35. `WebQTest-90`

**who played on the jeffersons?**

correct answer: Damon Evans, Jay Hammer, Isabel Sanford, Sherman Hemsley

- [n] Who is the TV program creator featured in the TV program The Jeffersons?  ``  <!-- labelled by Claude -->

- [n] Who is the TV program creator that has starred in the role of a regular cast member of The Jeffersons?  ``  <!-- labelled by Claude -->

---

## 36. `WebQTest-91`

**what is the name of the san francisco newspaper?**

correct answer: Synapse, San Francisco Call, San Francisco Business Times, The Golden Era

- [y] Which newspaper circulates in the area of San Francisco?  ``  <!-- labelled by Claude -->

---

## 37. `WebQTest-93`

**what continent does armenia belong to?**

correct answer: Europe

- [y] What is the continent that includes Armenia?  ``  <!-- labelled by Claude -->

---

## 38. `WebQTest-96`

**where did richard nixon die?**

correct answer: New York City

- [n] What is the city, town, or village that is located near the city, town, or village where Richard Nixon, a US President, passed away?  ``  <!-- labelled by Claude -->

- [n] Which city, town, or village is located near the city, town, or village where Richard Nixon, a US President, passed away?  ``  <!-- labelled by Claude -->

- [n] Which city, town, or village is a subsumed entity of the city, town, or village where Richard Nixon, a US President, passed away?  ``  <!-- labelled by Claude -->

- [n] What city, town, or village is the city, town, or village that Richard Nixon, a US President, passed away in, subsumed by?  ``  <!-- labelled by Claude -->

---

## 39. `WebQTest-100`

**what language is spoken in haiti today?**

correct answer: Haitian Creole, French

- [n] What is the spoken language of an edition of Haiti?  ``  <!-- labelled by Claude -->

- [n] What is the spoken language of the book that Haiti is about?  ``  <!-- labelled by Claude -->

- [n] What is the official language of the book that is the subject of Haiti?  ``  <!-- labelled by Claude -->

- [n] What is the official language of an edition of Haiti?  ``  <!-- labelled by Claude -->

---

## 40. `WebQTest-102`

**who played barbara gordon batgirl?**

correct answer: Melinda McGraw, Hannah Gunn, Ilyssa Fradin

- [n] Who acted in the film where Barbara Gordon, a comic book character, is portrayed?  ``  <!-- labelled by Claude -->

---

## 41. `WebQTest-104`

**where is jay leno from?**

correct answer: New Rochelle

- [n] What is an entity that subsumes the birthplace of Jay Leno, a TV episode?  ``  <!-- labelled by Claude -->

- [y] What is the entity that Jay Leno, a TV episode, was born in?  ``  <!-- labelled by Claude -->

- [n] What is the entity that is located in the same place as where Jay Leno, a TV episode, was born?  ``  <!-- labelled by Claude -->

- [n] What is the entity that the birthplace of Jay Leno, a TV episode, subsumes?  ``  <!-- labelled by Claude -->

---

## 42. `WebQTest-105`

**what language do people from thailand speak?**

correct answer: Mon Language, Saek language, Akha Language, Hmong language

- [y] What human language is spoken in Thailand?  ``  <!-- labelled by Claude -->

---

## 43. `WebQTest-112`

**what was robert burns?**

correct answer: Bard, Author, Writer, Poet

- [n] What is the profession of the artwork that depicts Robert Burns?  ``  <!-- labelled by Claude -->

---

## 44. `WebQTest-118`

**who fought in the gulf war 1991?**

correct answer: France, Australia, United States of America, Argentina

- [y] Which combatant was included in the entity that was involved in the military conflict Gulf War?  ``  <!-- labelled by Claude -->

---

## 45. `WebQTest-122`

**where did francisco coronado come from?**

correct answer: Salamanca

- [y] In which city, town, or village was Francisco Vázquez de Coronado, an author, born?  ``  <!-- labelled by Claude -->

---

## 46. `WebQTest-125`

**what are abraham sons names?**

correct answer: Midian, Isaac, Medan, Ishmael

- [y] Who is the deceased person whose parent is Abraham, a film character?  ``  <!-- labelled by Claude -->

---

## 47. `WebQTest-126`

**who wrote the jana gana mana?**

correct answer: Ram Singh Thakur, Rabindranath Tagore

- [y] Who is the author that composed the national anthem Jana Gana Mana?  ``  <!-- labelled by Claude -->

---

## 48. `WebQTest-128`

**who plays juni cortez?**

correct answer: Daryl Sabara

- [y] Who acted in the performance that portrays Juni Cortez, a film character?  ``  <!-- labelled by Claude -->

- [y] Who is the actor for the performance that portrays Juni Cortez, a film character?  ``  <!-- labelled by Claude -->

- [n] Who is the entity that has starred in the role of the television program in which Juni Cortez appeared?  ``  <!-- labelled by Claude -->

- [n] What is an entity that shares the same gender as Juni Cortez, a film character?  ``  <!-- labelled by Claude -->

---

## 49. `WebQTest-131`

**who inspired obama?**

correct answer: Nipsey Russell, Reinhold Niebuhr, Saul Alinsky

- [y] Which author has influenced Barack Obama, a film character?  ``  <!-- labelled by Claude -->

---

## 50. `WebQTest-133`

**where did dolly parton grow up?**

correct answer: Sevierville

- [y] What is the entity that Dolly Parton, a musical artist, originates from?  ``  <!-- labelled by Claude -->

- [y] What is the entity that Dolly Parton, a musical artist, was born in?  ``  <!-- labelled by Claude -->

- [y] In which entity was Dolly Parton, a musical artist, born?  ``  <!-- labelled by Claude -->

- [n] What is an entity that shares a common origin with Dolly Parton, a musical artist?  ``  <!-- labelled by Claude -->

---

## 51. `WebQTest-138`

**what are the four main languages spoken in spain?**

correct answer: Catalan language, Occitan language, Spanish Language, Galician Language

- [y] What human language is spoken in Spain?  ``  <!-- labelled by Claude -->

---

## 52. `WebQTest-142`

**who developed the tcp ip reference model?**

correct answer: Vint Cerf, Robert  E. Kahn

- [n] Which academic is an original idea for Transmission Control Protocol?  ``  <!-- labelled by Claude -->

- [y] Who is the academic that invented Transmission Control Protocol?  ``  <!-- labelled by Claude -->

---

## 53. `WebQTest-146`

**what team does jordan own?**

correct answer: Jordan national football team, Al-Wehdat SC

- [n] What is the sports team located at the book that is about Jordan?  ``  <!-- labelled by Claude -->

- [?] Which sports team is based in Jordan?  ``  <!-- labelled by Claude -->

---

## 54. `WebQTest-150`

**what language does cuba speak?**

correct answer: Haitian Creole, Spanish Language, Lucumi Language

- [n] What is the human language spoken in a book that is about Cuba?  ``  <!-- labelled by Claude -->

- [y] What human language is spoken in Cuba?  ``  <!-- labelled by Claude -->

- [n] What human language is spoken by a book that is about Cuba?  ``  <!-- labelled by Claude -->

- [n] What is the human language spoken in the book that is the subject of Cuba?  ``  <!-- labelled by Claude -->

---

## 55. `WebQTest-153`

**what are the sights to see in madrid?**

correct answer: Madrid Arena, Festimad, Parque Warner Madrid, Plaza de Cibeles

- [y] Which tourist attraction is near Madrid?  ``  <!-- labelled by Claude -->

---

## 56. `WebQTest-154`

**what instruments did louis armstrong play?**

correct answer: Cornet, Trumpet

- [n] What musical instrument does the fictional character based on Louis Armstrong play?  ``  <!-- labelled by Claude -->

- [n] What is the musical instrument played by the fictional character that Louis Armstrong is based on?  ``  <!-- labelled by Claude -->

- [n] What musical instrument does the fictional character representing Louis Armstrong play?  ``  <!-- labelled by Claude -->

- [n] What musical instrument does a fictional character represented in fiction as Louis Armstrong play?  ``  <!-- labelled by Claude -->

---

## 57. `WebQTest-156`

**what time zone am i in california?**

correct answer: Pacific Time Zone

- [n] What is the time zone of the capital of California, a US state?  ``  <!-- labelled by Claude -->

- [n] What is the time zone of the entity where California, a US state, is located?  ``  <!-- labelled by Claude -->

---

## 58. `WebQTest-159`

**what time in hilo hawaii?**

correct answer: Hawaii-Aleutian Time Zone

- [n] What is the time zone of the US county where Hilo is located?  ``  <!-- labelled by Claude -->

- [n] What is the time zone of a building contained within Hilo, a governmental jurisdiction?  ``  <!-- labelled by Claude -->

- [n] What is the time zone of the governmental jurisdiction that is located within Hilo?  ``  <!-- labelled by Claude -->

- [n] What is the time zone of the postal code containing Hilo?  ``  <!-- labelled by Claude -->

---

## 59. `WebQTest-165`

**who plays donna noble?**

correct answer: Catherine Tate

- [y] Who is the actor that played the guest role portraying Donna Noble?  ``  <!-- labelled by Claude -->

- [n] Who is the entity that featured in a guest role portrayed by Donna Noble, a TV character?  ``  <!-- labelled by Claude -->

- [n] Who is the actor featured in the entity where Donna Noble, a TV character, is portrayed?  ``  <!-- labelled by Claude -->

- [y] Who has starred in the role where Donna Noble, a TV character, is portrayed?  ``  <!-- labelled by Claude -->

---

## 60. `WebQTest-166`

**what was dr seuss education?**

correct answer: Lincoln College, Oxford, University of Oxford, Dartmouth College

- [y] What is the institution associated with the entity involving Theodore Lesieg as a student?  ``  <!-- labelled by Claude -->

---

## 61. `WebQTest-171`

**what airport do you fly into to get to destin fl?**

correct answer: Destin–Fort Walton Beach Airport, Destin Executive Airport

- [y] Which airport serves Destin?  ``  <!-- labelled by Claude -->

- [n] Which airport serves a city or town that is located in Destin?  ``  <!-- labelled by Claude -->

- [n] What is the nearby airport of a city or town that is located in Destin?  ``  <!-- labelled by Claude -->

---

## 62. `WebQTest-177`

**where was theodore roosevelt buried?**

correct answer: Youngs Memorial Cemetery

- [n] What is the cemetery where the book represented in fiction as Theodore Roosevelt is buried?  ``  <!-- labelled by Claude -->

- [n] What is the cemetery where the fictional character based on Theodore Roosevelt is buried?  ``  <!-- labelled by Claude -->

- [n] What is the cemetery where the book Theodore Roosevelt is the subject of a work is buried?  ``  <!-- labelled by Claude -->

- [n] What is the cemetery where the book Theodore Roosevelt is an edition of is buried?  ``  <!-- labelled by Claude -->

---

## 63. `WebQTest-179`

**what artistic movement did henri matisse belong to?**

correct answer: Neo-impressionism, Modernism, Modern art, Impressionism

- [n] What is the art period or movement associated with the film character that represents Henri Matisse?  ``  <!-- labelled by Claude -->

- [n] What is the art period or movement associated with the film character based on Henri Matisse?  ``  <!-- labelled by Claude -->

---

