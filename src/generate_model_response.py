import os
import re
import json
import time
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generate model responses for PL and RF tasks.")
    parser.add_argument('--model_type', type=str, required=True, choices=['MoE', 'FT', 'Base'], help='Type of model: MoE, FT, or Base')
    parser.add_argument('--task', type=str, required=True, choices=['pl', 'rf'], help='Task type: Predicate Label (pl) or Research Field (rf)')
    parser.add_argument('--model_id', type=str, required=True, help='Hugging Face model ID to load')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the input data file (pickle format)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum retries for failed generations (default: 3)')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum number of tokens to generate (default: 512)')
    parser.add_argument('--prompt_type', type=str, default='zero_shot', choices=['zero_shot', 'few_shot', 'cot', 'zero_shot_cot'], help='Prompt type: zero_shot, few_shot, cot, zero_shot_cot')
    parser.add_argument('--version', type=str, default='v1', help='Version identifier for the output')
    return parser.parse_args()

# Function to load model and tokenizer
def load_model_and_tokenizer(model_id):
    logger.info(f"Loading model and tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

# Function to create prompts
def create_prompt(task, prompt_type, title, abstract):
    if task == 'pl':
        if prompt_type == 'zero_shot':
            return f"""
            The "predicate_labels" categorizes key elements of a research paper, including its research problems, methods, results, materials, specific scientific metrics, and supplementary information, providing a structured overview of the paper's content and contributions.
            Below is the title and abstract of a research paper. Based on the information provided, determine the most appropriate predicate labels.Finally provide the response in the following JSON format:
            {{
                "paper_title": "Insert paper title here",
                "predicate_labels": [List of predicate labels here]
            }}

            Paper Title: {title}

            Abstract: {abstract}

            response in JSON format:
            '''
            """
        
        elif prompt_type == 'few_shot':
            return f"""
            The "predicate_labels" categorizes key elements of a research paper, including its research problems, methods, results, materials, specific scientific metrics, and supplementary information, providing a structured overview of the paper's content and contributions.
            Below is the title and abstract of a research paper. Based on the information provided, determine the most appropriate predicate labels.Finally provide the response in JSON format.

            Paper Title: Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation

            Abstract: When performing visually guided actions under conditions of perturbed visual feedback, e.g., in a mirror or a video camera, there is a spatial conflict between visual and proprioceptive information. Recent studies have shown that subjects without proprioception avoid this conflict and show a performance benefit. In this study, we tested whether deafferentation induced by repetitive transcranial magnetic stimulation (rTMS) can improve mirror tracing skills in normal subjects. Hand trajectory error during novel mirror drawing was compared across two groups of subjects that received either 1 Hz rTMS over the somatosensory cortex contralateral to the hand or sham stimulation. Mirror tracing was more accurate after rTMS than after sham stimulation. Using a position-matching task, we confirmed that rTMS reduced proprioceptive acuity and that this reduction was largest when the coil was placed at an anterior parietal site. It is thus possible, with rTMS, to enhance motor performance in tasks involving a visuoproprioceptive conflict, presumably by reducing the excitability of somatosensory cortical areas that contribute to the sense of hand position.

            response in JSON format:
            {{
                "paper_title": "Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation",
                "predicate_labels": ["stimulator company", "stimulation intensity selection approach", "threshold-estimation strategies", "threshold measurement", "stimulator model", "type of rTMS", "coil shape", "threshold-estimation strategies", "stimulator company", "stimulation intensity selection approach", "threshold measurement", "stimulator model", "type of rTMS", "coil shape"]
            }}

            Paper Title: Enteral Clostrid- ium difficile, an emerging cause for high-output ileostomy

            Abstract: The loss of fluid and electrolytes from a high-output ileostomy (&gt;1200 ml/day) can quickly result in dehydration and if not properly managed may cause acute renal failure. The management of a high-output ileostomy is based upon three principles: correction of electrolyte disturbance and fluid balance, pharmacological reduction of ileostomy output, and treatment of any underlying identifiable cause. There is an increasing body of evidence to suggest that <jats:italic>Clostridium difficile</jats:italic> may behave pathologically in the small intestine producing a spectrum of enteritis that mirrors the well-recognised colonic disease manifestation. Clinically this can range from high-output ileostomy to fulminant enteritis. This report describes two cases of high-output ileostomy associated with enteric <jats:italic>C difficile</jats:italic> infection and proposes that the management algorithm of a high-output ileostomy should include exclusion of small bowel <jats:italic>C difficile</jats:italic>.

            response in JSON format:
            {{
                "paper_title": "Enteral Clostridium difficile, an emerging cause for high-output ileostomy",
                "predicate_labels": ["research problem", "inflammatory bowel disease", "Endoscopy", "Intestinal operation", "Treatment"]
            }}

            Paper Title: Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China

            Abstract:  **Background**: In December 2019, the coronavirus disease 2019 (COVID-19), caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), broke out in Wuhan. While the epidemiological and clinical characteristics of patients with COVID-19 have been reported, the relationships between laboratory features and viral load have not been comprehensively described. **Methods**: Adult inpatients (≥18 years old) with COVID-19 who underwent multiple nucleic acid tests (≥5 times) with nasal and pharyngeal swabs were recruited from Renmin Hospital of Wuhan University. The study included general patients (n = 70), severe patients (n = 195), and critical patients (n = 43). Laboratory, demographic, and clinical data were extracted from electronic medical records. A fitted polynomial curve was used to explore the association between serial viral loads and illness severity. **Results**: Viral load of SARS-CoV-2 peaked within the first 2–4 days after admission and then decreased rapidly, with a virus rebound under treatment. Critical patients had the highest viral loads, while general patients showed the lowest. Viral loads were higher in sputum compared to nasal and pharyngeal swabs (P = 0.026). The positive rate of respiratory tract samples was significantly higher than gastrointestinal tract samples (P < 0.001). The SARS-CoV-2 viral load was negatively correlated with certain blood routine parameters and lymphocyte subsets, and positively associated with laboratory features of the cardiovascular system. **Conclusions**: Serial viral loads during hospitalization revealed viral shedding patterns and resurgence during treatment. These findings could be used as early warning indicators for illness severity and to improve antiviral interventions.

            response in JSON format:
            {{
                "paper_title": "Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China",
                "predicate_labels": ["Study type", "Location", "research problem", "Time period for data collection", "material", "Statistical tests", "method", "result", "patient age"]
            }}
            
            Now determine the most appropriate predicate labels for this new paper only and provide the final response in JSON format:
            
            Paper Title: {title}

            Abstract: {abstract}

            response in JSON format:
            """
        
        elif prompt_type == 'cot':
            return f"""
            The "predicate_labels" categorizes key elements of a research paper, including its research problems, methods, results, materials, specific scientific metrics, and supplementary information, providing a structured overview of the paper's content and contributions.
            Below is the title and abstract of a research paper. Based on the information provided, determine the most appropriate predicate labels.

            Paper Title: "Title of the Paper"
            Abstract: "Abstract of the Paper"

            Step-by-Step Reasoning:
            1. **Identify the key focus of the paper**: Determine the main research area or topic from the title and abstract.
            2. **Analyze the methodology or approach**: Understand the method or approach used in the study (e.g., specific techniques, tools, or strategies mentioned).
            3. **Examine the research problem or objective**: Identify the main problem or goal the study addresses.
            4. **Evaluate the results or findings**: Review the outcomes, findings, or observations from the study.
            5. **Classify additional aspects**: Consider any other experimental conditions, locations, time periods, or materials.
            6. **Generate the appropriate predicate label(s)**: Based on this reasoning, select the predicate labels that describe the action or relationship of the study.

            Please provide the response in the following JSON format:
            {{
                "paper_title": "Insert paper title here",
                "predicate_labels": [List of predicate labels here]
            }}

            Paper Title: Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation  
            
            Abstract: When performing visually guided actions under conditions of perturbed visual feedback, e.g., in a mirror or a video camera, there is a spatial conflict between visual and proprioceptive information. Recent studies have shown that subjects without proprioception avoid this conflict and show a performance benefit. In this study, we tested whether deafferentation induced by repetitive transcranial magnetic stimulation (rTMS) can improve mirror tracing skills in normal subjects. Hand trajectory error during novel mirror drawing was compared across two groups of subjects that received either 1 Hz rTMS over the somatosensory cortex contralateral to the hand or sham stimulation. Mirror tracing was more accurate after rTMS than after sham stimulation. Using a position-matching task, we confirmed that rTMS reduced proprioceptive acuity and that this reduction was largest when the coil was placed at an anterior parietal site. It is thus possible, with rTMS, to enhance motor performance in tasks involving a visuoproprioceptive conflict, presumably by reducing the excitability of somatosensory cortical areas that contribute to the sense of hand position.  

            Step-by-Step Reasoning:
            1. Key Focus: The paper focuses on the effects of repetitive transcranial magnetic stimulation (rTMS) on mirror drawing accuracy by affecting proprioception.
            2. Methodology: The study uses rTMS to induce proprioceptive deafferentation and compares two groups with different stimulation types.
            3. Research Problem: It aims to explore whether reducing proprioception enhances motor performance in visual-motor conflict tasks.
            4. Results: Mirror tracing is more accurate after rTMS, suggesting performance improvement in visuoproprioceptive tasks.
            5. Additional Aspects: The study uses a position-matching task to confirm the effects of rTMS on proprioception.
            6. Predicate Label: The focus is on **stimulator company**, **stimulation intensity selection approach**, and **threshold-estimation strategies** due to the technical aspects of rTMS.

            response in JSON format:
            {{
                "paper_title": "Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation",
                "predicate_labels": ["stimulator company", "stimulation intensity selection approach", "threshold-estimation strategies"]
            }}

            Paper Title: Enteral Clostrid- ium difficile, an emerging cause for high-output ileostomy

            Abstract: The loss of fluid and electrolytes from a high-output ileostomy (&gt;1200 ml/day) can quickly result in dehydration and if not properly managed may cause acute renal failure. The management of a high-output ileostomy is based upon three principles: correction of electrolyte disturbance and fluid balance, pharmacological reduction of ileostomy output, and treatment of any underlying identifiable cause. There is an increasing body of evidence to suggest that <jats:italic>Clostridium difficile</jats:italic> may behave pathologically in the small intestine producing a spectrum of enteritis that mirrors the well-recognised colonic disease manifestation. Clinically this can range from high-output ileostomy to fulminant enteritis. This report describes two cases of high-output ileostomy associated with enteric <jats:italic>C difficile</jats:italic> infection and proposes that the management algorithm of a high-output ileostomy should include exclusion of small bowel <jats:italic>C difficile</jats:italic>.

            Step-by-Step Reasoning:
            1. Key Focus: The paper focuses on the clinical management of high-output ileostomy caused by Clostridium difficile infection. It highlights how C. difficile can affect the small intestine, leading to complications that require specific treatment strategies.

            2. Methodology: The study reports two clinical cases where high-output ileostomy was associated with C. difficile infection. The management includes balancing fluids, correcting electrolytes, and addressing the underlying infection.

            3. Research Problem: The primary research problem is identifying Clostridium difficile as an emerging cause of high-output ileostomy in the small intestine, expanding the understanding of its pathological role beyond colonic disease.

            4. Results: The report shows that C. difficile infection can lead to significant complications, such as dehydration and renal failure, in patients with high-output ileostomy, suggesting a need to include C. difficile testing in the management protocol for such cases.

            5. Additional Aspects: The paper discusses three key principles for managing high-output ileostomy: correcting fluid and electrolyte balance, pharmacologically reducing output, and treating the underlying cause, specifically targeting C. difficile infection.

            6. Predicate Labels: The paper addresses the research problem of managing high-output ileostomy related to C. difficile. It also connects to inflammatory bowel disease, discusses the role of endoscopy in diagnosis, mentions intestinal operation as relevant to the condition, and outlines treatment strategies.

            response in JSON format:
            {{
                "paper_title": "Enteral Clostridium difficile, an emerging cause for high-output ileostomy",
                "predicate_labels": ["research problem", "inflammatory bowel disease", "Endoscopy", "Intestinal operation", "Treatment"]
            }}

            Paper Title: Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China

            Abstract:  **Background**: In December 2019, the coronavirus disease 2019 (COVID-19), caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), broke out in Wuhan. While the epidemiological and clinical characteristics of patients with COVID-19 have been reported, the relationships between laboratory features and viral load have not been comprehensively described. **Methods**: Adult inpatients (≥18 years old) with COVID-19 who underwent multiple nucleic acid tests (≥5 times) with nasal and pharyngeal swabs were recruited from Renmin Hospital of Wuhan University. The study included general patients (n = 70), severe patients (n = 195), and critical patients (n = 43). Laboratory, demographic, and clinical data were extracted from electronic medical records. A fitted polynomial curve was used to explore the association between serial viral loads and illness severity. **Results**: Viral load of SARS-CoV-2 peaked within the first 2–4 days after admission and then decreased rapidly, with a virus rebound under treatment. Critical patients had the highest viral loads, while general patients showed the lowest. Viral loads were higher in sputum compared to nasal and pharyngeal swabs (P = 0.026). The positive rate of respiratory tract samples was significantly higher than gastrointestinal tract samples (P < 0.001). The SARS-CoV-2 viral load was negatively correlated with certain blood routine parameters and lymphocyte subsets, and positively associated with laboratory features of the cardiovascular system. **Conclusions**: Serial viral loads during hospitalization revealed viral shedding patterns and resurgence during treatment. These findings could be used as early warning indicators for illness severity and to improve antiviral interventions.

            Step-by-Step Reasoning:
            1. Key Focus: The paper focuses on viral shedding patterns in COVID-19 patients and their relationship to illness severity.
            2. Methodology: Multiple nucleic acid tests (nasal and pharyngeal swabs) were conducted on adult COVID-19 inpatients.
            3. Research Problem: The study aims to understand the association between viral load and clinical severity of COVID-19.
            4. Results: Higher viral loads were seen in critical patients, with sputum samples showing higher viral load than nasal or pharyngeal swabs.
            5. Additional Aspects: The study took place in Wuhan and collected data over multiple days using clinical and laboratory records.
            6. Predicate Labels: The study involves study type, location, research problem, time period for data collection, material, statistical tests, method, result, and patient age.

            response in JSON format:
            {{
            "paper_title": "Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China",
            "predicate_labels": ["Study type", "Location", "research problem", "Time period for data collection", "material", "Statistical tests", "method", "result", "patient age"]
            }}

            Now determine the most appropriate predicate labels for this new paper only and provide the final response in JSON format:
            Paper Title: {title}
            Abstract: {abstract}

            Step-by-Step Reasoning:
            """
        
        elif prompt_type == 'zero_shot_cot':
            return f"""
            The "predicate_labels" categorizes key elements of a research paper, including its research problems, methods, results, materials, specific scientific metrics, and supplementary information, providing a structured overview of the paper's content and contributions.
            Below is the title and abstract of a research paper. Based on the information provided, determine the most appropriate predicate labels.Finally provide the response in JSON format.
            
            Paper Title: "Title of the Paper"
            Abstract: "Abstract of the Paper"

            Step-by-Step Reasoning:
            1. **Identify the key focus of the paper**: Determine the main research area or topic from the title and abstract.
            2. **Analyze the methodology or approach**: Understand the method or approach used in the study (e.g., specific techniques, tools, or strategies mentioned).
            3. **Examine the research problem or objective**: Identify the main problem or goal the study addresses.
            4. **Evaluate the results or findings**: Review the outcomes, findings, or observations from the study.
            5. **Classify additional aspects**: Consider any other experimental conditions, locations, time periods, or materials.
            6. **Generate the appropriate predicate label(s)**: Based on this reasoning, select the predicate labels that describe the action or relationship of the study.

            Please provide the response in the following JSON format:
            {{
                "paper_title": "Insert paper title here",
                "predicate_label": [List of predicate labels here]
            }}

            Now determine the most appropriate predicate labels for this new paper only and provide the final response in JSON format:

            Paper Title: {title}
            Abstract: {abstract}

            Step-by-Step Reasoning:
            """
        



    elif task == 'rf':
        if prompt_type == 'zero_shot':
            return f"""
            Below is the title and abstract of a research paper. Based on the information provided, determine the most appropriate single research field or category to which the paper belongs. Finally provide the response in the following JSON format:
            {{
                "paper_title": "Insert paper title here",
                "research_field": "Research field label here"
            }}

            Paper Title: {title}

            Abstract: {abstract}

            response in JSON format:
            """
        
        elif prompt_type == 'few_shot':
            return f"""
            Below is the title and abstract of a research paper. Based on the information provided, determine the most appropriate single research field or category to which the paper belongs.

            1. Paper Title: Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation

            Abstract: When performing visually guided actions under conditions of perturbed visual feedback, e.g., in a mirror or a video camera, there is a spatial conflict between visual and proprioceptive information. Recent studies have shown that subjects without proprioception avoid this conflict and show a performance benefit. In this study, we tested whether deafferentation induced by repetitive transcranial magnetic stimulation (rTMS) can improve mirror tracing skills in normal subjects. Hand trajectory error during novel mirror drawing was compared across two groups of subjects that received either 1 Hz rTMS over the somatosensory cortex contralateral to the hand or sham stimulation. Mirror tracing was more accurate after rTMS than after sham stimulation. Using a position-matching task, we confirmed that rTMS reduced proprioceptive acuity and that this reduction was largest when the coil was placed at an anterior parietal site. It is thus possible, with rTMS, to enhance motor performance in tasks involving a visuoproprioceptive conflict, presumably by reducing the excitability of somatosensory cortical areas that contribute to the sense of hand position.

            Research Field Label: Computational Neuroscience

            response in JSON format:
            {{
                "paper_title": "Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation",
                "research_field": "Computational Neuroscience"
            }}

            2. Paper Title: Enteral Clostrid- ium difficile, an emerging cause for high-output ileostomy

            Abstract: The loss of fluid and electrolytes from a high-output ileostomy (&gt;1200 ml/day) can quickly result in dehydration and if not properly managed may cause acute renal failure. The management of a high-output ileostomy is based upon three principles: correction of electrolyte disturbance and fluid balance, pharmacological reduction of ileostomy output, and treatment of any underlying identifiable cause. There is an increasing body of evidence to suggest that <jats:italic>Clostridium difficile</jats:italic> may behave pathologically in the small intestine producing a spectrum of enteritis that mirrors the well-recognised colonic disease manifestation. Clinically this can range from high-output ileostomy to fulminant enteritis. This report describes two cases of high-output ileostomy associated with enteric <jats:italic>C difficile</jats:italic> infection and proposes that the management algorithm of a high-output ileostomy should include exclusion of small bowel <jats:italic>C difficile</jats:italic>.

            Research Field Label: Science

            response in JSON format:
            {{
                "paper_title": "Enteral Clostridium difficile, an emerging cause for high-output ileostomy",
                "research_field": "Science"
            }}

            3. Paper Title: Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China

            Abstract: <jats:title>Abstract</jats:title><jats:sec><jats:title>Background</jats:title>In December 2019, the coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) broke out in Wuhan. Epidemiological and clinical characteristics of patients with COVID-19 have been reported, but the relationships between laboratory features and viral load has not been comprehensively described.</jats:sec><jats:sec><jats:title>Methods</jats:title>Adult inpatients (≥18 years old) with COVID-19 who underwent multiple (≥5 times) nucleic acid tests with nasal and pharyngeal swabs were recruited from Renmin Hospital of Wuhan University, including general patients (n = 70), severe patients (n = 195), and critical patients (n = 43). Laboratory data, demographic data, and clinical data were extracted from electronic medical records. The fitted polynomial curve was used to explore the association between serial viral loads and illness severity.</jats:sec><jats:sec><jats:title>Results</jats:title>Viral load of SARS-CoV-2 peaked within the first few days (2–4 days) after admission, then decreased rapidly along with virus rebound under treatment. Critical patients had the highest viral loads, in contrast to the general patients showing the lowest viral loads. The viral loads were higher in sputum compared with nasal and pharyngeal swab (P = .026). The positive rate of respiratory tract samples was significantly higher than that of gastrointestinal tract samples (P &amp;lt; .001). The SARS-CoV-2 viral load was negatively correlated with portion parameters of blood routine and lymphocyte subsets and was positively associated with laboratory features of cardiovascular system.</jats:sec><jats:sec><jats:title>Conclusions</jats:title>The serial viral loads of patients revealed whole viral shedding during hospitalization and the resurgence of virus during the treatment, which could be used for early warning of illness severity, thus improve antiviral interventions.</jats:sec>

            Research Field Label: Virology

            response in JSON format:
            {{
                "paper_title": "Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China",
                "research_field": "Virology"
            }}

            Now determine the most appropriate single research field or category for this new paper only and provide the final response in JSON format:

            Paper Title: {title}
            
            Abstract: {abstract}

            response in JSON format:
            """
        

        elif prompt_type == 'cot':
            return f"""
            Below is the title and abstract of a research paper. Based on the information provided, think step by step to determine the most appropriate single research field for the paper.
            
            Paper Title: "Title of the Paper"
            Abstract: "Abstract of the Paper"

            Step-by-Step Reasoning:
            1. **Identify the main topic or subject**: Based on the title and abstract, determine the core area of research the paper is focused on.
            2. **Analyze the research methods**: Look at the research techniques or methodologies used (e.g., clinical trials, experiments, data analysis) and understand what kind of field they apply to.
            3. **Examine the type of research**: Consider whether the research is theoretical, experimental, clinical, or something else.
            4. **Evaluate the purpose and results**: Determine the goal of the research and how the findings relate to specific disciplines or fields.
            5. **Determine the subfields, if applicable**: Identify if the paper belongs to any subfields or interdisciplinary areas.
            6. **Generate the research field**: Based on this reasoning, determine a single research field.

            Please provide the response in the following JSON format:
            {{
                "paper_title": "Insert paper title here",
                "research_field": "Research field label"
            }}

            Paper Title: Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation
            
            Abstract: When performing visually guided actions under conditions of perturbed visual feedback, e.g., in a mirror or a video camera, there is a spatial conflict between visual and proprioceptive information. Recent studies have shown that subjects without proprioception avoid this conflict and show a performance benefit. In this study, we tested whether deafferentation induced by repetitive transcranial magnetic stimulation (rTMS) can improve mirror tracing skills in normal subjects. Hand trajectory error during novel mirror drawing was compared across two groups of subjects that received either 1 Hz rTMS over the somatosensory cortex contralateral to the hand or sham stimulation. Mirror tracing was more accurate after rTMS than after sham stimulation. Using a position-matching task, we confirmed that rTMS reduced proprioceptive acuity and that this reduction was largest when the coil was placed at an anterior parietal site. It is thus possible, with rTMS, to enhance motor performance in tasks involving a visuoproprioceptive conflict, presumably by reducing the excitability of somatosensory cortical areas that contribute to the sense of hand position.

            Step-by-Step Reasoning:

            1. Main Topic: The paper is focused on the effects of transcranial magnetic stimulation on proprioception and motor performance during visually guided tasks.
            2. Research Methods: The study uses repetitive transcranial magnetic stimulation (rTMS) to examine its effects on motor control and proprioception.
            2. Type of Research: This is experimental neuroscience research focusing on motor systems and sensory feedback.
            3. Results and Purpose: The goal is to explore whether proprioceptive deafferentation improves motor performance in visual-proprioceptive conflict tasks.
            4.Research Field Label: The study fits into Computational Neuroscience as it deals with neural processes, motor control, and sensory systems using rTMS.

            response in JSON format:
            {{
                "paper_title": "Enhanced Accuracy in Novel Mirror Drawing after Repetitive Transcranial Magnetic Stimulation-Induced Proprioceptive Deafferentation",
                "research_field": "Computational Neuroscience"
            }}

            Paper Title: Enteral Clostrid- ium difficile, an emerging cause for high-output ileostomy

            Abstract: The loss of fluid and electrolytes from a high-output ileostomy (&gt;1200 ml/day) can quickly result in dehydration and if not properly managed may cause acute renal failure. The management of a high-output ileostomy is based upon three principles: correction of electrolyte disturbance and fluid balance, pharmacological reduction of ileostomy output, and treatment of any underlying identifiable cause. There is an increasing body of evidence to suggest that <jats:italic>Clostridium difficile</jats:italic> may behave pathologically in the small intestine producing a spectrum of enteritis that mirrors the well-recognised colonic disease manifestation. Clinically this can range from high-output ileostomy to fulminant enteritis. This report describes two cases of high-output ileostomy associated with enteric <jats:italic>C difficile</jats:italic> infection and proposes that the management algorithm of a high-output ileostomy should include exclusion of small bowel <jats:italic>C difficile</jats:italic>.

            Step-by-Step Reasoning:
            1. Main Topic: The paper focuses on the clinical management of high-output ileostomy, particularly when it is caused by Clostridium difficile infection in the small intestine. The study aims to shed light on how C. difficile affects the small intestine, a lesser-known manifestation of this pathogen.

            2. Research Methods: The paper presents two case studies of patients with high-output ileostomy associated with C. difficile infection. The management approach includes balancing fluids and electrolytes, reducing ileostomy output through medication, and treating the underlying infection.

            3. Type of Research: This is clinical and case-based research, focusing on medical case management and the pathological role of C. difficile beyond its well-known effects in the colon. The paper emphasizes understanding the broad spectrum of enteritis caused by the infection.

            4. Results and Purpose: The study concludes that Clostridium difficile infection in the small intestine can lead to severe complications like dehydration and renal failure. It suggests modifying the management algorithm for high-output ileostomy to include testing for small bowel C. difficile infection, emphasizing the importance of recognizing this emerging cause.

            5. Research Field Label: The research falls under the broad field of Science since it addresses medical and microbiological aspects of C. difficile infections, focusing on clinical implications and treatment strategies in a medical context.

            response in JSON format:
            {{
                "paper_title": "Enteral Clostridium difficile, an emerging cause for high-output ileostomy",
                "research_field": "Science"
            }}

            Paper Title: Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China

            Abstract: <jats:title>Abstract</jats:title><jats:sec><jats:title>Background</jats:title>In December 2019, the coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) broke out in Wuhan. Epidemiological and clinical characteristics of patients with COVID-19 have been reported, but the relationships between laboratory features and viral load has not been comprehensively described.</jats:sec><jats:sec><jats:title>Methods</jats:title>Adult inpatients (≥18 years old) with COVID-19 who underwent multiple (≥5 times) nucleic acid tests with nasal and pharyngeal swabs were recruited from Renmin Hospital of Wuhan University, including general patients (n = 70), severe patients (n = 195), and critical patients (n = 43). Laboratory data, demographic data, and clinical data were extracted from electronic medical records. The fitted polynomial curve was used to explore the association between serial viral loads and illness severity.</jats:sec><jats:sec><jats:title>Results</jats:title>Viral load of SARS-CoV-2 peaked within the first few days (2–4 days) after admission, then decreased rapidly along with virus rebound under treatment. Critical patients had the highest viral loads, in contrast to the general patients showing the lowest viral loads. The viral loads were higher in sputum compared with nasal and pharyngeal swab (P = .026). The positive rate of respiratory tract samples was significantly higher than that of gastrointestinal tract samples (P &amp;lt; .001). The SARS-CoV-2 viral load was negatively correlated with portion parameters of blood routine and lymphocyte subsets and was positively associated with laboratory features of cardiovascular system.</jats:sec><jats:sec><jats:title>Conclusions</jats:title>The serial viral loads of patients revealed whole viral shedding during hospitalization and the resurgence of virus during the treatment, which could be used for early warning of illness severity, thus improve antiviral interventions.</jats:sec>

            Step-by-Step Reasoning:
            1. Main Topic: The paper is focused on COVID-19, particularly the viral shedding patterns of SARS-CoV-2 in adult patients. It explores how viral load changes during hospitalization and how these changes correlate with illness severity.

            2. Research Methods: The study used multiple nucleic acid tests (nasal and pharyngeal swabs) to measure viral loads in different patient groups (general, severe, and critical). It employed statistical modeling (fitted polynomial curves) to analyze the relationship between viral load and clinical severity.

            3. Type of Research: This is an experimental and clinical research paper, focusing on the laboratory analysis of viral loads and their correlation with clinical outcomes in patients with COVID-19. It combines virology with clinical epidemiology.

            4. Results and Purpose: The findings show that viral load peaks within a few days of admission and is higher in critically ill patients. The study provides insights into viral shedding patterns, which can be used to predict illness severity and guide antiviral interventions.

            5. Research Field Label: The paper clearly falls within Virology, as it focuses on the behavior of the SARS-CoV-2 virus in the human body, specifically its shedding patterns and correlation with disease severity.

            response in JSON format:
            {{
                "paper_title": "Chronological Changes of Viral Shedding in Adult Inpatients with COVID-19 in Wuhan, China",
                "research_field": "Virology"
            }}

            Now determine the most appropriate single research field for this new paper only and provide the final response in JSON format:

            Paper Title: {title}
            Abstract: {abstract}

            Step-by-Step Reasoning:
            """
        
        elif prompt_type == 'zero_shot_cot':
            return f"""
            Below is the title and abstract of a research paper. Based on the information provided, think step by step to determine the most appropriate single research field for the paper.
            
            Paper Title: "Title of the Paper"
            Abstract: "Abstract of the Paper"

            Step-by-Step Reasoning:
            1. **Identify the main topic or subject**: Based on the title and abstract, determine the core area of research the paper is focused on.
            2. **Analyze the research methods**: Look at the research techniques or methodologies used (e.g., clinical trials, experiments, data analysis) and understand what kind of field they apply to.
            3. **Examine the type of research**: Consider whether the research is theoretical, experimental, clinical, or something else.
            4. **Evaluate the purpose and results**: Determine the goal of the research and how the findings relate to specific disciplines or fields.
            5. **Determine the subfields, if applicable**: Identify if the paper belongs to any subfields or interdisciplinary areas.
            6. **Generate the research field**: Based on this reasoning, determine the most appropriate single research field.

            Please provide the response in the following JSON format:
            {{
                "paper_title": "Insert paper title here",
                "research_field": "Research field label"
            }}

            Now determine the most appropriate single research field for this new paper only and provide the final response in JSON format:

            Paper Title: {title}
            Abstract: {abstract}

            Step-by-Step Reasoning:
            """
    raise ValueError(f"Unsupported task or prompt type: {task}, {prompt_type}")

# Function to extract information from the model response
def extract_response(response_text, task):
    try:
        if task == 'pl':
            match = re.search(r'Predicate Labels:\s*(\[.*?\])', response_text, re.DOTALL)
            return json.loads(match.group(1)) if match else []
        elif task == 'rf':
            match = re.search(r'Research Field:\s*"(.*?)"', response_text)
            return match.group(1).strip() if match else None
    except Exception as e:
        logger.error(f"Failed to extract response: {e}")
        return None

# Function to generate responses
def generate_responses(model, tokenizer, df, task, prompt_type, max_tokens, max_retries):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    responses = []
    logger.info("Generating responses...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title = row['title']
        abstract = row['abstract']
        prompt = create_prompt(task, prompt_type, title, abstract)

        for attempt in range(max_retries):
            try:
                response = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)[0]['generated_text']
                extracted = extract_response(response, task)
                responses.append((row['title'], response, extracted))
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for row {idx}: {e}")
                time.sleep(2)  # Delay before retry
        else:
            logger.error(f"Failed to generate response for row {idx} after {max_retries} retries.")
            responses.append((row['title'], None, None))

    return responses

# Main function
def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # Load data
    logger.info(f"Loading data from {args.data_file}")
    df = pd.read_pickle(args.data_file)
    logger.info("Data loaded successfully.")

    # Generate responses
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        df=df,
        task=args.task,
        prompt_type=args.prompt_type,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries
    )

    # Save results
    output_file = os.path.join(args.output_dir, f"{args.model_type}_{args.task}_{args.version}_responses.pkl")
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(responses, columns=['title', 'raw_response', 'extracted']).to_pickle(output_file)
    logger.info(f"Responses saved to {output_file}")

if __name__ == "__main__":
    main()
