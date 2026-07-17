# SNU AI CHALLENGE HOMEPAGE

## ABOUT

### Overview

The Graduate School of Data Science at Seoul National University conducts active research at the forefront of artificial intelligence, with the aim of overcoming the limitations of existing models and improving generalization performance across diverse situations. We also make significant efforts to disseminate these research outcomes and encourage more people to take an interest in AI research. As part of these efforts, we are releasing a dataset processed by our graduate school and providing an opportunity for undergraduate students from universities in Korea and abroad to develop state-of-the-art AI models and compete based on their performance.

### Challenge Description

#### 1. Task Definition: Reconstructing Scenes from Text

Participants must solve a task in which four image frames are rearranged into the correct order according to a given storyline or caption.

introduction

#### 2. Background and Significance

This task evaluates multimodal understanding beyond recognizing images and captions independently. Participants must reconstruct multiple scenes within the context of a storyline or caption and restore the correct temporal progression.

#### 3. Input and Output Format

A brief description of the input and output format is provided below.

- **Input:** Data consisting of a natural-language sentence and multiple frames  
  Example: `{ "text": "A natural-language sentence", "frames": [image_3, image_1, image_4, image_2] }`
- **Output:** The position of each provided frame after the frames are rearranged into the correct order  
  Example: `[3, 4, 1, 2]` means that the first provided frame is third in the correct sequence, the second provided frame is fourth, and so on.

#### 4. Competition Schedule

- Early registration opens: June 15, 2026
- Preliminary round (online competition): June 29–July 24, 2026
- Final round (public presentation evaluation): August 7, 2026

> Registration closes on July 17. All teams wishing to participate must register before the deadline.

Please refer to the **Timeline** tab for the detailed schedule.

#### 5. Competition Process

The competition consists of a preliminary round and a final round.

During the preliminary round, participants develop models using the provided training data and submit inference results for the test data. The leaderboard is updated using only 70% of the full test set, referred to as the public data.

After the preliminary round, the highest-ranked teams based on performance over the full test set will undergo code verification and report review. Preliminary scores will be finalized after this process, and approximately ten teams will advance to the final round.

Depending on the level of performance achieved by participating teams, an additional evaluation based on an external dataset may be conducted after the preliminary round.

The final round will be conducted as an offline presentation evaluation. Participants will present the model-development process from the preliminary round to the judges. Final winners will be determined by considering both the quantitative and qualitative scores achieved during the preliminary round and the final-presentation score.

| Category | Description | Points |
|---|---|---:|
| Preliminary-round performance | Inference accuracy on the preliminary test data; logical validity, efficiency, and originality of the task-solving strategy, evaluated through the written report | 40 |
| Data utilization | Appropriateness of data preprocessing and utilization strategies | 15 |
| Model design and training methodology | Selection and application of a model architecture and training methodology appropriate for the task | 15 |
| Optimization methodology | Degree of model-performance optimization under inference-environment constraints | 10 |
| Resource efficiency | Memory usage and response speed or latency required to perform the task | 10 |
| Development cost | Computational cost for training and preprocessing costs, including external API usage | 10 |
| **Total** |  | **100** |

#### 6. Awards and Benefits

Total prize pool: **KRW 23,000,000**

| Award | Division | Number of Teams | Prize |
|---|---|---:|---:|
| Seoul National University President's Award | Grand Prize | 1 | KRW 10,000,000 |
| Gold Prize | Gold | 1 | KRW 5,000,000 |
| Silver Prize | Silver | 1 | KRW 3,000,000 |
| Bronze Prize | Bronze | 1 | KRW 2,000,000 |
| Encouragement Prize | Merit | 3 | KRW 1,000,000 each |

#### 7. Competition Rules

Please refer to the **Rules** tab. Failure to comply with the rules may result in disqualification.

#### 8. Eligibility

- Undergraduate students who are currently enrolled in or on an approved leave of absence from a university in Korea or abroad
- No restrictions on major or year of study
- Prize money can only be deposited into a Korean bank account
- Please contact the organizers for additional questions regarding eligibility

## TIMELINE

| Date and Time (KST) | Event |
|---|---|
| June 22–July 17 | Registration |
| June 29, 10:00 AM | Competition begins |
| June 29, 10:00 AM–July 24, 11:59 PM | Preliminary round and submission period |
| July 17 | Registration closes |
| July 25 | Final leaderboard released using public and private data |
| July 25–28 | Top-ranked teams submit verification code and reports |
| August 3 | Finalists announced |
| August 6 | Final-presentation materials due |
| August 7, subject to change | Final presentation evaluation |
| Late August | Final rankings announced and awards presented |

All deadlines, including the deadline for preliminary inference submissions and verification-code submissions, are **11:59 PM Korea Standard Time on the final day of the stated period**.

## DATA

### 1. Dataset Overview

This competition focuses on predicting the temporal order of video frames. Given four shuffled video-frame images and a corresponding text description, or `Sentence`, participants must reconstruct the correct chronological order of the original video.

The dataset is divided into training and test sets. Each sample has a unique `Id`, and the corresponding image folder and metadata are provided in CSV files. No separate validation dataset is provided. Participants may reserve part of the training set for validation.

The source videos used to extract the frames were collected from various sources. Some frames were generated using generative AI. The text corresponding to each video frame was either written by humans or generated with generative AI.

The training set contains unrefined data. In some cases, the correct answer may not be uniquely identifiable from the frames alone, or frames unrelated to the text, such as black frames, may be included. Effective preprocessing of such samples may improve model performance and is included in the final-round evaluation criteria.

The target label lists the original position of each provided frame. For example, if the original order is `a, b, c, d` and the shuffled order is `a, d, b, c`, the answer is `[1, 4, 2, 3]`.

### 2. Task Examples

Examples from the competition are shown below.

data

data

data

### 3. Data Access and Use

The dataset may only be used for participation in this competition. Redistribution of the original dataset or any modified version is prohibited.

## RULES

### 1. Eligibility

- Undergraduate students currently enrolled in or on an approved leave of absence from a university in Korea or abroad may participate. There are no restrictions on major or year of study.
- Prize money can only be paid into a Korean bank account under the team representative's name.
- A certificate of enrollment issued within one month of the competition start date must be submitted with the application.
- Please contact the organizers with additional eligibility questions.

### 2. Participation Rules

- Participants may enter individually or as a team of up to four members.
- Teams must be formed at the time of registration. Teams may not be merged or split after registration. Violations will result in disqualification.
- Registering or participating through multiple accounts is prohibited. If detected, the entire team will be disqualified.

### 3. Model Training, Inference, and External Resources

#### 3.1 Execution Environment

- Required programming language: Python
- Every model must be executable in an offline local environment using a CPU or GPU.

#### 3.2 External API Usage

- External commercial APIs such as ChatGPT, Gemini, or Grok may not be used during model training or inference.
- External commercial APIs may be used only for data preprocessing, and the total cost may not exceed KRW 30,000.
- When an external API is used, the presentation materials must state how it was used, the experimental results, and the incurred cost.
- A team may be disqualified if the total development cost is determined to exceed KRW 30,000.

#### 3.3 Data and Models

- External training data is prohibited. Only the training data provided by the competition may be used.
- Open-source models such as LLaMA may be used only when their model weights were publicly released on or before May 31, 2026.
- When eligibility is unclear, participants must ask the organizers in advance through the Kaggle Discussion tab.
- Model ensembles, meaning combinations of inference outputs from multiple models, are prohibited. This also includes splitting the dataset, fine-tuning the same model separately on each split, and combining the resulting predictions.
- Data augmentation using the provided data is allowed, but generating or transforming data with a generative model is prohibited.
- Model-compression techniques such as quantization and LoRA are permitted.
- Inference strategies and intermediate reasoning methods, including Chain-of-Thought, multi-turn chat, and test-time augmentation, may be used as long as they comply with the inference-time limit.
- Inference over the entire test set must finish within 24 hours in the verification environment described in Section 4.

#### 3.4 Data Leakage

The use of information from the evaluation dataset during training is prohibited.

Examples include:

- Manually labeling evaluation data and using those labels for training
- Analyzing test-set characteristics and using that analysis to design preprocessing or the model

Violations will result in disqualification.

#### 3.5 Rule Violations

Failure to comply with the rules may result in disqualification. During the competition, high-ranked teams may be asked to provide code and results for reproducibility verification.

### 4. Submission Rules

- Submissions are limited to two per day.
- The final submitted model must run on a single NVIDIA RTX 3090 GPU with 24 GB of VRAM.

Verification-server specifications:

- CPU: AMD EPYC 7502 32-Core Processor × 2
- Memory: 512 GB
- GPU: GeForce RTX 3090
- NVIDIA driver: 550.54.15
- CUDA: 12.4

### 5. Preliminary-Round Evaluation

- During the preliminary round, the leaderboard is updated using 70% of the full test set, referred to as the public data.
- Rankings are determined using Exact Match Accuracy.
- A prediction is counted as correct only when the entire image order exactly matches the target. If even one position differs, the sample is counted as incorrect.
- For example, when the answer is `[1, 4, 2, 3]`, only `[1, 4, 2, 3]` is accepted as correct.
- After the preliminary round, the highest-ranked teams based on the full public and private test set will undergo code verification and report review.
- Depending on the overall performance achieved, a separate evaluation using an external dataset may be conducted.

### 6. Code Reproducibility Verification and Report Submission

After the preliminary round, participants ranked within the top 16 must submit the following materials within the designated period:

- Training and inference code in `.py` format
- Final model-weight files
- A methodology report of no more than five A4 pages, submitted as a Microsoft Word document; no fixed template is provided

Submitted code must meet the following requirements:

- Provide a detailed `README.md`
- Use relative paths for data input and output
- Use UTF-8 encoding for code and comments
- Ensure that all code executes without errors
- Specify the development environment, including the operating system, hardware, and library versions
- Provide a separate inference script and the model-weight files used by it
- The total size of the model-execution code and weights must not exceed 80 GB

Detailed procedures will be announced separately.

### 7. Final-Presentation Materials

Teams with the highest preliminary scores that pass code-reproducibility verification will participate in an offline presentation evaluation.

- Presentation materials must be submitted in PPT or PDF format by the day before the presentation.
- Presentations must be no longer than ten minutes.

### 8. Final-Presentation Rules

- Presentation: 10 minutes
- Q&A: 5 minutes
- The organizers will prepare the previously submitted presentation file before the session.
- In principle, one representative from each team will present.
- All team members must attend the final-presentation event.
- Presentations will be open to the public.

> Additional rules may be introduced. Participants should check for updates regularly to avoid penalties.  
> Last updated: July 1, 12:00 PM

## PARTICIPATION

This competition is conducted on the Kaggle platform.

Participants should complete the Google Form using the registration link below.

Registration link

After registration, the organizers will verify eligibility and grant access to the Kaggle competition. Access is processed in batches at 11:30 AM on weekdays, with an additional batch at 6:00 PM on Fridays. No processing is conducted on weekends; weekend applications are processed on Monday.

Kaggle allows team formation only after a participant has made at least one submission. After receiving access, participants should first make one submission and then use the **Team** tab to form their official team.

Kaggle competition link

## FAQ

The following section summarizes frequently asked questions. For additional questions, please contact the organizers.

**Q. I am an undergraduate student at a university outside Korea. May I participate?**  
Yes. Undergraduate students enrolled at overseas universities are eligible. However, prize money can only be paid into a Korean bank account.

**Q. May high-school or graduate students participate?**  
No. This competition is limited to undergraduate students. If even one member of a team is not an undergraduate student, the entire team will be disqualified.

**Q. I have completed my undergraduate coursework but have not graduated. May I participate?**  
Yes, provided that you can submit an official document, such as a certificate of enrollment or registration status, proving that you have not graduated.

**Q. How are teams formed?**  
Participants may enter individually or as a team of up to four members. Teams must be formed at the time of registration and cannot be merged after registration. Participation through multiple accounts is prohibited, and detection will result in disqualification of the entire team.

**Q. May external data be used in addition to the provided training data?**  
No. The use of any external dataset not provided by the competition is prohibited.

**Q. Are there restrictions on which models may be used?**  
Yes. Models must satisfy all of the following conditions:

- The model must have been publicly released as open source on or before May 31, 2026.
- It must run locally without an internet connection.
- External commercial APIs such as ChatGPT, Gemini, and Grok may not be used during training or inference.
- External commercial APIs may be used only for preprocessing, in which case the total cost must not exceed KRW 30,000.

When the rules are ambiguous, please ask through the competition discussion board and refer to the Rules tab.

**Q. Is there a submission limit?**  
Yes. Each participant or team may submit up to twice per day, based on the UTC period from 00:00 to 23:59.

**Q. Must the submission file be a CSV file?**  
Yes. Submissions must follow the required CSV format and encoding. Please refer to the Rules page for details such as column names, row count, sorting requirements, and encoding.

**Q. Where will the final round be held?**  
The final round will be held at Seoul National University. The detailed time and venue will be announced later.

**Q. Who should give the final presentation?**  
In principle, one representative from each team should present.

**Q. Must every team member attend the final presentation?**  
Yes. All team members must attend.

**Q. Who evaluates the final presentations?**  
Faculty members from the Seoul National University Graduate School of Data Science will serve as judges.

**Q. The final model must run on an NVIDIA RTX 3090 with 24 GB of VRAM. How can I verify this without owning that GPU?**  
After the preliminary round, ranked participants will be provided access to an NVIDIA RTX 3090 environment so that they can verify model setup and execution. Refer to Section 4, Submission Rules, in the Rules tab for detailed server specifications.

**Q. May I publish a paper based on the methodology used in this competition?**  
The intellectual-property rights and permitted use of competition outputs are governed by the competition rules and may include rights held by the organizers. Participants considering publication should contact the organizers in advance.

**Q. How is prize money divided within a team?**  
The entire prize will be paid to the person registered as the team representative. Team members should decide internally how to divide it.

**Q. Are prizes subject to tax?**  
Yes, taxes may apply. Tax obligations are the responsibility of the prize recipient.

# KAGGLE PAGE

## OVERVIEW

### Overview

The Graduate School of Data Science at Seoul National University conducts active research at the forefront of artificial intelligence, with the aim of overcoming the limitations of existing models and improving generalization performance across diverse situations. We also work to disseminate these research outcomes and encourage broader interest in AI research. As part of these efforts, we are releasing a dataset processed by our graduate school and providing an opportunity for undergraduate students from universities in Korea and abroad to develop state-of-the-art AI models and compete based on performance.

**Start:** 24 days ago  
**Close:** 14 days remaining

### Description

#### 1. Task Definition: Reconstructing Scenes from Text

Participants must rearrange four image frames into the correct order according to a given storyline or caption.

#### 2. Background and Significance

This task evaluates multimodal understanding beyond recognizing images and captions independently. Participants must reconstruct multiple scenes within the context of a storyline or caption and restore the correct temporal progression.

#### 3. Input and Output Format

- **Input:** Data consisting of a natural-language sentence and multiple frames  
  Example: `{ "text": "A natural-language sentence", "frames": [image_3, image_1, image_4, image_2] }`
- **Output:** The position of each provided frame after rearranging the frames into the correct order  
  Example: `[3, 4, 1, 2]` means that the first provided frame is third in the correct sequence, the second is fourth, and so on.

#### 4. Competition Schedule

- Early registration opens: June 15, 2026
- Preliminary round: June 29–July 24, 2026
- Final presentation evaluation: August 7, 2026
- Registration closes: July 17, 2026

#### 5. Competition Process

The competition consists of a preliminary round and a final round.

During the preliminary round, participants develop models using the provided training data and submit inference results for the test data. The leaderboard is updated using only 70% of the full test set, referred to as the public data.

After the preliminary round, top-ranked teams based on performance over the full test set will undergo code verification and report review. Approximately ten teams will advance to the final round after scores are finalized.

Depending on participant performance, an additional evaluation using an external dataset may be conducted after the preliminary round.

The final round will be an offline presentation evaluation. Participants will present their model-development process to the judges. Final winners will be determined by combining quantitative and qualitative preliminary-round results with the final-presentation score.

| Category | Description | Points |
|---|---|---:|
| Model performance | Preliminary-test inference accuracy; logical validity, efficiency, and originality of the task-solving strategy, evaluated through the written report | 40 |
| Data utilization | Appropriateness of preprocessing and data-utilization strategies | 15 |
| Model design and training methodology | Selection and application of a model architecture and training methodology appropriate for the task | 15 |
| Optimization methodology | Model-performance optimization under inference-environment constraints | 10 |
| Resource efficiency | Memory usage and response speed or latency | 10 |
| Development cost | Training-compute and preprocessing costs, including external API usage | 10 |
| **Total** |  | **100** |

#### 6. Competition Rules

Please refer to the **Rules** tab. Failure to comply may result in disqualification.

#### 7. Eligibility

Undergraduate students currently enrolled in or on an approved leave of absence from a university in Korea or abroad may participate. There are no restrictions on major or year of study.

- **Host:** Seoul National University
- **Organizer:** Seoul National University Graduate School of Data Science
- **Sponsors:** Moreh Co., Ltd., Motif Technology Co., Ltd., and BK21

### Evaluation

Preliminary rankings are determined using Exact Match Accuracy. A prediction is considered correct only when the entire submitted image order exactly matches the answer. No partial credit is awarded. For example, when the answer is `[1, 4, 2, 3]`, only `[1, 4, 2, 3]` is accepted as correct.

## DATA

### Dataset Description

This competition focuses on predicting the temporal order of video frames. Given four shuffled frame images and a corresponding text description, or `Sentence`, participants must restore the correct chronological order of the original video.

The dataset is divided into training and test sets. Each sample has a unique `Id`, and the corresponding image folder and metadata are provided in CSV format.

### Files

- `train.csv`: Training data containing frame filenames, text descriptions, and target labels representing the original order
- `test.csv`: Evaluation data for which frame order must be predicted; target labels are not provided
- `sample_submission.csv`: Example submission format
- `train/`: Training-image folders; each folder name matches an `Id` in `train.csv`
- `test/`: Test-image folders; each folder name matches an `Id` in `test.csv`

### CSV Columns

#### `train.csv`

- `Id`: A unique six-character identifier made up of uppercase letters, lowercase letters, and digits
- `Sentence`: A text caption describing the video clip and providing clues for temporal-order inference
- `Input_1`–`Input_4`: Filenames of the four input frames
- Filenames contain randomized three-letter codes and do not include ordering information
- Images are provided in alphabetical filename order, which is unrelated to temporal order
- `No_ordering`: Indicates whether frame rearrangement is unnecessary. When `True`, the frames were not shuffled and the answer is fixed as `[1, 2, 3, 4]`
- `Answer`: Target label representing the original temporal position of each frame
- Format: a string representation of a four-element list, `[n, n, n, n]`, where each `n` is an integer from 1 to 4
- Example: `[2, 4, 3, 1]` means `Input_1` is the second frame in the original sequence, `Input_2` is fourth, and so on

#### `test.csv`

The test set follows a similar structure but excludes answer-related information.

- `Id`: Unique video-sample identifier
- `Sentence`: Text caption for the video
- `Input_1`–`Input_4`: Encoded image filenames

### Directory Structure

```text
train/
├── 00aB12/
│   ├── 00aB12_aek.jpg  (Input_1)
│   ├── 00aB12_bmw.jpg  (Input_2)
│   ├── 00aB12_cyd.jpg  (Input_3)
│   └── 00aB12_dqa.jpg  (Input_4)
├── 01cD34/
│   ├── ...
...

test/
├── 02eF56/
│   ├── ...
...
```

The dataset may only be used for participation in this competition. Redistribution of the original dataset or any modified version is prohibited.

## RULES

### Competition Rules

#### 1. Eligibility

- Undergraduate students currently enrolled in or on an approved leave of absence from a university in Korea or abroad may participate. There are no restrictions on major or year of study.
- Prize money can only be paid into a Korean bank account under the team representative's name.
- A certificate of enrollment issued within one month of the competition start date must be submitted with the application.
- Please contact the organizers with additional eligibility questions.

#### 2. Participation Rules

- Participants may enter individually or as a team of up to four members.
- Teams must be formed at registration and may not be merged or split afterward. Violations will result in disqualification.
- Registering through multiple accounts is prohibited. If detected, the entire team will be disqualified.

#### 3. Model Training, Inference, and External Resources

##### 3.1 Execution Environment

- Programming language: Python
- All models must run in an offline local CPU or GPU environment.

##### 3.2 External API Usage

- External commercial APIs such as ChatGPT, Gemini, and Grok may not be used during training or inference.
- External commercial APIs may be used only for data preprocessing, and the total cost may not exceed KRW 30,000.
- Presentation materials must disclose the method of API use, experimental results, and incurred cost.
- A team may be disqualified if total development cost is determined to exceed KRW 30,000.

##### 3.3 Data and Models

- External training data is prohibited. Only the provided training data may be used.
- Open-source models such as LLaMA may be used only when their model weights were publicly released on or before May 31, 2026.
- When model eligibility is unclear, participants must ask the organizers in advance through the Kaggle Discussion tab.
- Model ensembles are prohibited. This includes splitting the dataset, fine-tuning the same model separately on each split, and combining the inference results.
- Model-compression techniques such as quantization and LoRA are permitted.
- Inference strategies and intermediate reasoning approaches, including Chain-of-Thought, multi-turn chat, and test-time augmentation, are permitted provided they comply with the time limit.
- Inference over the full test set must finish within 24 hours in the verification environment described in Section 4.

##### 3.4 Data Leakage

Using evaluation-dataset information for training is prohibited.

Examples include:

- Manually labeling evaluation data and using it for training
- Analyzing evaluation-data characteristics and using the findings for preprocessing or model design

Violations will result in disqualification.

##### 3.5 Rule Violations

Failure to comply with the rules may result in disqualification. High-ranked participants may be asked to provide code and results for reproducibility verification during the competition.

#### 4. Submission Rules

- Submissions are limited to two per day.
- The final model must run on one NVIDIA RTX 3090 GPU with 24 GB of VRAM.

Server specifications:

- CPU: AMD EPYC 7502 32-Core Processor × 2
- Memory: 512 GB
- GPU: GeForce RTX 3090
- NVIDIA driver: 550.54.15
- CUDA: 12.4

#### 5. Preliminary-Round Evaluation

- The public leaderboard is updated using 70% of the full test set.
- Rankings are determined using Exact Match Accuracy.
- A prediction is correct only when the complete image order exactly matches the answer.
- For example, when the answer is `[1, 4, 2, 3]`, only that exact list is accepted.
- After the preliminary round, top teams based on performance over the entire public and private test set will undergo code verification and report review.
- Depending on overall performance, an additional evaluation using an external dataset may be conducted.

#### 6. Code Reproducibility Verification and Report Submission

After the preliminary round, participants ranked within the top 16 must submit:

- Training and inference code in `.py` format
- Final model-weight files
- A methodology report of up to five A4 pages as a Microsoft Word document; no fixed template is provided

Requirements for submitted code:

- Include a detailed `README.md`
- Use relative paths for data input and output
- Use UTF-8 encoding for source code and comments
- Ensure that all code executes without errors
- State the development environment, including OS, hardware, and library versions
- Provide separate inference code and the corresponding model-weight files
- The total size of the model code and weights must not exceed 80 GB

Detailed procedures will be announced separately.

#### 7. Final-Presentation Materials

- Top-ranked teams that pass reproducibility verification will participate in an offline presentation evaluation.
- Presentation materials must be submitted in PPT or PDF format by the day before the presentation.
- Presentations must be no longer than ten minutes.

#### 8. Final-Presentation Rules

- Presentation: 10 minutes
- Q&A: 5 minutes
- The organizers will prepare the previously submitted presentation file.
- In principle, one representative from each team will present.
- All team members must attend.
- Presentations will be public.

> Additional rules may be introduced. Participants should check for updates regularly.  
> Last updated: June 29, 7:00 PM


Kaggle Competition Foundational Rules
(Non-editable)

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.
The following Kaggle Competition Foundational Rules (“ Foundational Rules ”) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize. b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records. c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible. d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website. b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website. c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules. d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification. b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code. c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set. b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email. b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria. c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph. d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle. e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded. f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants. b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions). c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8. d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed. e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above. f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>. b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law. b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity. c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition. c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard. d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission. e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition. f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition. g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules. h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor. i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.

