IBM Debater (R): Evidence Sentences (version 1.0)
-------------------------------------------------

The corpus contains 5,785 annotated pairs of a topic and a sentence.
It has been split into 4,066 pairs for train and 1,719 for test.
83 topics are represented in train and 35 topics in test, with no topic overlap.
Train contains 1,499 positives, test contains 683 positives.

This file is only the README, the data set itself is available on the
IBM Debater Datasets webpage: http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml
and is comprised of 3 files: this readme, train.csv and test.csv.


Structure of the CSV files
--------------------------

Each CSV file presents 6 columns:

1. topic
		The debating topic serving as context for the sentence candidate.
2. the concept of the topic
		The main concept featured in the topic, as obtained through wikification.
3. candidate
		The sentence candidate. Extracted from Wikipedia, but cleaned of Wikisyntax,
		with footnote markers replaced with a "[REF]" tag.
4. candidate masked
		The sentence candidate, with any surface form of the Wiki concept of the topic
		replaced with a "TOPIC_CONCEPT" mask. Again obtained through wikification.
		This is the column we used both for training and for test as we want the network 
		to be indifferent to the topic.
5. label
		1 for evidence (either supporting or contesting the topic), 0 for non-evidence
6. wikipedia article name
		The title of the Wikipedia article where the candidate was found.
7. wikipedia url
		The address of the Wikipedia page featuring the article containing the candidate.

		
Examples
--------

Here are 3 lines taken from train.csv, following the line format:
topic,the concept of the topic,candidate,candidate masked,label,wikipedia article name,wikipedia url

We should fight illegal immigration,illegal immigration,"Professor of Law Francine Lipman [REF] writes that the belief that illegal migrants are exploiting the US economy and that they cost more in services than they contribute to the economy is ""undeniably false"".","Professor of Law Francine Lipman [REF] writes that the belief that TOPIC_CONCEPT are exploiting the US economy and that they cost more in services than they contribute to the economy is ""undeniably false"".",1,Economic impact of illegal immigrants in the United States,https://en.wikipedia.org/wiki/Economic_impact_of_illegal_immigrants_in_the_United_States
We should further exploit wind turbines,wind turbines,The Guyana Energy Agency reported a pilot project for a wind turbine in Guyana's east coast.,The Guyana Energy Agency reported a pilot project for TOPIC_CONCEPT in Guyana's east coast.,0,https://en.wikipedia.org/wiki/Electricity_sector_in_Guyana
We should legalize cannabis,cannabis,"On 23 May 2006, Donald Tashkin, M.D., Professor of Medicine at the David Geffen School of Medicine at UCLA in Los Angeles announced that the use of cannabis does not appear to increase the risk of developing lung cancer, or increase the risk of head and neck cancers, such as cancer of the tongue, mouth, throat, or esophagus [REF].","On 23 May 2006, Donald Tashkin, M.D., Professor of Medicine at the David Geffen School of Medicine at UCLA in Los Angeles announced that the use of TOPIC_CONCEPT does not appear to increase the risk of developing lung cancer, or increase the risk of head and neck cancers, such as cancer of the tongue, mouth, throat, or esophagus [REF].",1,Cannabis-associated respiratory disease,https://en.wikipedia.org/wiki/Long-term_effects_of_cannabis


Annotation process
------------------

Each topic-sentence pair was annotated by 10 crowd-sourced labelers. The majority label was assigned (with a 50/50 tie being considered non-evidence).

The guidelines provided to the annotators present mainly 3 criteria, which all have to be met for a positive label.

1. The sentence must clearly support or contest the topic, and not simply be neutral.
2. It has to be coherent and stand mostly on its own.
3. It has to be convincing, something you could use to sway someone's stance on the topic (a claim is not enough, it has to be backed up).

The annotators' agreement is 0.45 by Fleiss’ kappa. This is a typical value in such challenging labeling tasks. In addition, for 85% of the labeled instances, the majority vote included at least 70% of the annotators.


Origin of the candidates
------------------------

Those sentences were identified as candidates in past experiments. Using a proprietary massive corpus, 
labeled by crowd, which we cannot shared due to legal restrictions, we trained a high-quality model
for evidence detection. We used the predictions of that model over Wikipedia to collect candidates
for annotation, over which a relatively high fraction is expected to represent positive examples
(rather than generating candidates by taking random sentences). Importantly, the only use of the model
in this work was to select candidates for labeling.


Pre-processing
--------------

These pre-processing steps have been applied to each sentence in the data set:

1) The Wikipedia raw text is cleaned from any Wikisyntax save for footnote markers,
which are replaced with a common token [REF]: knowing that a statement which provides
its source can be an indication of quality argumentation.

2) In the masked version of the candidates, every occurrence of the main concept of the topic
in the candidate is masked: this allows to train argument mining systems to be topic-independent.
To implement this masking we applied wikification – the detection and linking to Wikipedia entities –
on both the topic and the evidence candidate. We used an in-house wikification tool (upcoming publication),
and there are such tools freely available, such as TagMe (Ferragina and Scaiella, 2010).
The words of the candidate that were wikified to the same Wikipedia title as the one to
which the topic concept was wikified, were replaced with the common token TOPIC_CONCEPT.

3) The sentences have passed a number of filters before being considered for annotation.
They are first filtered by length: at least 7 tokens and at most 50. The average length of a sentence in the corpus 
is 31 tokens (same for positives and negatives), with a lexicon size of 14K different tokens.
A number of manually crafted rules were then applied to ensure candidates coherency. They do not start
with a pronoun nor end with a question mark. They also do not contain an unresolved he/she pronoun,
or a contraction -- e.g. introduced by however -- as these often indicate a missing context.


Licensing and copyright
-----------------------

The dataset is released under the licensing and copyright terms mentioned in the
IBM Debater Datasets webpage: http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml.
  

Citation
--------
           
If you use the data, please cite the following publication

   Will it Blend? Blending Weak and Strong Labeled Data in a Neural Network for Argumentation Mining
   Eyal Shnarch, Carlos Alzate, Lena Dankin, Martin Gleize, Yufang Hou, Leshem Choshen, Ranit Aharonov and Noam Slonim.
   Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018


List of the 83 train topics
---------------------------

We should limit executive compensation
We should ban partial birth abortions
Homeschooling should be banned
We should increase government regulation
We should adopt libertarianism
We should subsidize adoptions
We should fight illegal immigration
We should fight gender inequality
We should abandon coal mining
We should increase gun control
We should ban full-body scanners
We should end development aid
We should adopt mobile payments
We should abandon Gmail
We should abolish temporary employment
We should abandon online dating services
We should legalize organ trade
We should legalize cannabis
We should legalize cell phone use while driving
We should abolish the Olympic Games
Sex education should be mandatory
We should abolish intellectual property rights
We should further exploit geothermal energy
We should further exploit solar energy
Bullfighting should be banned
We should further exploit wind power
We should end international aid
We should ban breast implants
We should abolish the right to keep and bear arms
We should further exploit wind turbines
We should legalize prostitution
We should ban corporal punishment in the home
We should increase wealth redistribution
We should support a phase-out of lightweight plastic bags
We should subsidize condoms
National service in the United States should be mandatory
Suicide should be a criminal offence
We should ban male infant circumcision
We should adopt a zero tolerance policy in schools
We should introduce a pollution tax
We should ban free newspapers
We should subsidize biofuels
We should lower the drinking age
We should ban fishing
We should legalize insider trading
Physical education should be mandatory
We should increase the use of personalized medicine
We should adopt direct democracy
The vow of celibacy should be abandoned
We should privatize the United States Social Security
We should end affirmative action
We should legalize polygamy
We should limit the right of asylum
We should abolish the three-strikes laws
We should prohibit hydraulic fracturing
We should introduce school vouchers
We should ban strip clubs
We should adopt atheism
We should ban trans fats usage in food
We should further exploit hydroelectric dams
We should not subsidize cultivation of tobacco
We should abolish the attorney-client privilege
We should prohibit tower blocks
We should adopt open source software
Private universities should be banned
Abstinence-only sex education should be mandatory
We should introduce covenant marriage
We should prohibit international adoption
We should introduce universal health care
We should prohibit corporal punishment
We should increase internet censorship
We should adopt socialism
Holocaust denial should be a criminal offence
We should introduce a flat tax
We should abolish zoos
We should cancel the speed limit
We should fight for Tibetan independence
We should prohibit reality television
We should ban private education
We should subsidize student loans
We should adopt vegetarianism
We should ban boxing
We should abolish the monarchy


List of the 35 test topics
--------------------------

We should limit the Internet of things
We should ban the sale of violent video games to minors
We should ban naturopathy
We should ban Piercing and Tattoos for minors
We should legalize the growing of coca leaf
We should end censorship
We should abolish anti-social behavior orders
We should abolish electronic voting
We should abolish homework
We should adopt multiculturalism
We should increase ecotourism
We should ban gambling
We should limit genetic testing
We should abolish the double jeopardy defense
We should further exploit nuclear power
We should not subsidize single parents
We should limit the freedom of speech
We should fight for Palestinian independence
The free market should be protected
We should abolish marriage
We should end mining
We should legalize same sex marriage
We should subsidize recycling
We should legalize ivory trade
We should ban human cloning
We should prohibit flag burning
We should ban fraternities
Big governments should be abandoned
We should adopt blasphemy laws
We should introduce compulsory voting
We should abandon disposable diapers
We should support water privatization
We should protect endangered species
We should legalize doping in sport
We should subsidize public service broadcasters