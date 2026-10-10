## Project brief
This capstone synthesises every concept from Week 4 into a single end-to-end deliverable. You will curate a domain dataset, fine-tune LLaMA 3.1 8B with QLoRA on Vast.AI, merge the adapter, run local inference, evaluate against the base model on held-out questions, and write a business recommendation a stakeholder can act on. The pipeline is exactly the one Olu ran this week; the domain is yours to choose.

## Compute responsibility
Your fine-tuning job is the most resource-intensive task of the week and you are responsible for your own vast.ai credits. The moment your training run completes, stop the instance from the vasi.ai console. Do not leave it running overnight or after your evaluation is done.

## Choose your domain
Domain stakeholder context suggested dataset focus
    1) Healthcare operations - (AfyaPlus) Community health platform in peri-urban Kenya	Appointment workflows, triage escalation SOPs, registration, system access; strictly non-diagnostic, always deferring clinical judgement to a provider
    2) Kenyan agricultural advisory - Agritech startup serving smallholder farmers	Pest control, soil management, crop-specific advice for Kenyan counties
    3) Kenyan legal aid - Legal aid NGO providing basic legal information	Land rights, tenancy, labour law, basic procedure under Kenyan law
    4) SME financial advisory - Microfinance institution serving Kenyan SMEs	Loan eligibility, business registration, tax obligations, mobile money integration
    5) Custom domain - Propose your own	Must have a clear Kenyan context and meet the curation standards below.

## Standards for every domain, including custom ones
1) Data sourcing: your curated examples must come from authoritative, domain-specific materials such as government manuals, certified training documents, or verified professional guidelines. 

2) Quality: examples must be accurate, consistent, and structured in the LLaMA instruction-tuning JSONL format; avoid synthetic or web-scraped content without rigorous manual verification. 

3) Contextual relevance: every domain needs a clear Kenyan grounding, such as specific national regulations, county practices, or local financial rules. Safety and ethics: 

NOTE: regardless of domain, your model must not give high-stakes advice such as diagnoses, binding legal judgments, or financial guarantees; build the appropriate disclaimer into your system prompt and your validation gate.

## Required deliverables
#### Deliverable 1: Dataset (20 marks)
At least 100 curated examples in the LLaMA instruction-tuning JSONL format, sourced from authoritative materials rather than generated; a validation report showing zero errors and an acceptable token range; a train/validation/test split in the correct 80/10/10 proportions; and a 300-word curation note explaining how examples were sourced, what quality and safety criteria were applied, and what coverage gaps remain.

#### Deliverable 2: Training run (25 marks)
Your completed fine_tune.py with every hyperparameter documented and justified; the trainer_state.json from the completed run; a loss curve analysis with a screenshot or plot of the training and validation progression; and a written diagnosis of whether the run was healthy, overfit, or underfit, using the Tuesday patterns.

#### Deliverable 3: Merged model and inference (20 marks)
merge_model.py completed and the merged model saved; local_inference.py running successfully with at least 5 sample responses to domain queries; and the verification gate passed, including your domain's mandatory disclaimer appearing in the responses.

#### Deliverable 4: Evaluation report (20 marks)
The evaluation pipeline run on all 20 test examples; a comparison table covering ROUGE-L, LLM judge, and groundedness for base versus fine-tuned; a per-question breakdown identifying the 3 questions with the biggest improvement and the 3 with the smallest, with your explanation of why; and comparison_results.csv included in the submission.

#### Deliverable 5: Stakeholder memo (15 marks)
A one-page maximum recommendation addressed to your domain's decision-maker. It must summarise what was built in one jargon-free paragraph, present the quality improvement in plain language with percentages, state the compute cost of the run, give two specific next actions with business rationale, and identify one risk (such as hallucination) together with its mitigation (such as human-in-the-loop review).

## Grading rubric
Criterion	Full marks (80-100%)	Partial (50-79%)	Insufficient (0-49%)
Dataset (20)	100+ valid examples, validation passes with zero errors, all splits correctly sized, curation note explains sources and gaps	75-99 examples or validation has minor warnings; splits approximately correct	Fewer than 75 examples, structural errors, or wrong format
Training run (25)	Script complete, training executed successfully, loss curve plotted and correctly diagnosed, trainer_state.json present	Script complete but training not successfully executed, or run completed with no monitoring of validation loss	Script incomplete or training not attempted
Merge and inference (20)	Merge complete, inference produces relevant domain responses, verification checklist fully passed	Merge complete but inference responses generic or off-topic	Merge not completed
Evaluation (20)	All 20 test examples evaluated, comparison table complete, per-question analysis identifies meaningful patterns	Fewer than 15 examples evaluated or comparison table has gaps	Evaluation not run, or only one model evaluated
Stakeholder memo (15)	Clear, jargon-free, every claim backed by numbers from the evaluation, specific next actions with rationale	Partially clear but contains unexplained technical terms, or next actions lack rationale	Not written, or a copy of the technical evaluation report
Submission
Submit a single GitHub repository link (or a zip upload) containing all scripts, the dataset files, comparison_results.csv, and the stakeholder memo. Include test.jsonl in the repository: the grader verifies your evaluation results against it. Include a README.md following the template below so your project is reproducible end to end. Deadline: end of the week following this session.

README.md
# [Your Domain] Fine-Tuned Assistant

## Overview
A brief description of your fine-tuned LLaMA 3.1 8B assistant and the chosen domain.

## Project Structure
- data/: train.jsonl, val.jsonl, test.jsonl
- data/ raw curated dataset plus data_prep.py: formatting and validation
- fine_tune.py: training script (run on Nebius)
- merge_model.py: adapter merge
- local_inference.py: inference pipeline with safety filter
- evaluate_models.py: base vs fine-tuned evaluation
- comparison_results.csv: evaluation metrics
- memo.md: stakeholder recommendation

## How to Reproduce
1. Environment setup: pip install -r requirements.txt
2. Data: place your curated dataset in the project root, then python data_prep.py
3. Fine-tuning: run python fine_tune.py on your vast.ai instance (inside tmux)
4. Merging: download the adapter, place it in the project root, run python merge_model.py
5. Evaluation: python evaluate_models.py to generate comparison_results.csv

## Disclaimer
This model provides non-diagnostic operational guidance only. [Adapt to your domain.]
What these commands do: 
The README is graded implicitly through every deliverable: if the grader cannot reproduce your run from these steps, the affected deliverable cannot earn full marks.

## NCA-GENL alignment
This capstone exercises the certification's transfer learning and fine-tuning domain end to end: PEFT configuration, dataset curation, loss interpretation, merging, and evaluation. Completing it well is the strongest exam preparation in the course.

Rubric
Week 4 Capstone Project Rubric
Week 4 Capstone Project Rubric
Criteria	Ratings	Pts
This criterion is linked to a learning outcomeDataset
20 Pts
Full Marks (80-100%)
100+ valid examples, validation passes with zero errors, all splits correctly sized, curation note explains sources and gaps
12 Pts
Partial Marks (50-79%)
75-99 examples or validation has minor warnings; splits approximately correct
0 Pts
Insufficient (0-49%)
Fewer than 75 examples, structural errors, or wrong format
20 pts
This criterion is linked to a learning outcomeTraining Run
25 Pts
Full Marks (80-100%)
Script complete, training executed successfully, loss curve plotted and correctly diagnosed, trainer_state.json present
15 Pts
Partial Marks (50-79%)
Script complete but training not successfully executed, or run completed with no monitoring of validation loss
0 Pts
Insufficient (0-49%)
Script incomplete or training not attempted
25 pts
This criterion is linked to a learning outcomeMerge and Inference
20 Pts
Full Marks (80-100%)
Merge complete, inference produces relevant domain responses, verification checklist fully passed
12 Pts
Partial Marks (50-79%)
Merge complete but inference responses generic or off-topic
0 Pts
Insufficient (0-49%)
Merge not completed
20 pts
This criterion is linked to a learning outcomeEvaluation
20 Pts
Full Marks (80-100%)
All 20 test examples evaluated, comparison table complete, per-question analysis identifies meaningful patterns
12 Pts
Partial Marks (50-79%)
Fewer than 15 examples evaluated or comparison table has gaps
0 Pts
Insufficient (0-49%)
Evaluation not run, or only one model evaluated
20 pts
This criterion is linked to a learning outcomeStakeholder Memo
15 Pts
Full Marks (80-100%)
Clear, jargon-free, every claim backed by numbers from the evaluation, specific next actions with rationale
9 Pts
Partial Marks (50-79%)
Partially clear but contains unexplained technical terms, or next actions lack rationale
0 Pts
Insufficient (0-49%)
Not written, or a copy of the technical evaluation report
15 pts
Total points: 100