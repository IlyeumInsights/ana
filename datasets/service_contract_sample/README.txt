Ilyeum Service Contracts Dataset Sample
Youssef Bendraou, Nathan Ramoly and Baba Seydou Bamba 

This folder carries a sample of data used in the experiments presented in the paper "Toward a Generic AutoML-based Assistant for Contracts Negotiation" (2020).
It represents a sample of three contracts that were anonymized. The other contracts used in the experiments are not disclosed for confidentiality reasons.
Contracts are annotated per clause and sentence by an expert from previous negotiation and according to acceptance policies.
This dataset is provided after the generation of new samples.
It contains:
- texts/: text files carrying sentences of clauses
- clauses_label.csv: annotation of clauses
- sentences_label.csv: annotation of sentences of clauses
- policy_rules.swrl: example of policies rules

----------------------------------------------
texts/

This folder regroups all clauses decomposed as sentences in text files.
Text files names are formalised as follows: "clauseid_sentid.txt" where "clauseid" is a clause identifier and "sentit" is the identifier of the sentence within the clause.
As new clause samples were generated, some sentences are duplicated or share similar roots.

----------------------------------------------
clauses_label.csv

Carries the raw label of each clause. A clause is textually represented by each of its sentences.
- column 1: clause id, matching clauseid in text file name.
- column 2: invalidity type label (false, imprecise, missing, or none)
- column 3: violated policy label, that corresponds to a policy id.
- column 4: clause type label.

----------------------------------------------
sentences_label.csv

Carries the raw label of each sentence of the clauses.
- column 1: sentence id, matching a text file name.
- column 2: invalidity type label (false, imprecise, missing, or none)
- column 3: violated policy label, that corresponds to a policy id.

----------------------------------------------
policy_rules.swrl

Sample of simple (https://www.w3.org/Submission/SWRL/) rules that models acceptance policies.
These rules are analyzed in order to orient and to restrict the AutoML process. 
Note that when evaluated, values, in particular words ( such as notContains(?c, "preavis") ), are extended to synonyms. In other words, propositions are not evaluated for a single value, but multiple ones of the same family.
