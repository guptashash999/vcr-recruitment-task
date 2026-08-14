                                    Visual Commonsense Reasoning (VCR) using Qwen2.5-VL
--->Visual Commonsense Reasoning (VCR) using Qwen2.5-VL
This project is a solution for a recruitment task requiring a multimodal AI system that can look at an image, answer a question about it, and select the best explanation (rationale) for its answer.

Instead of just recognizing objects, this AI understands social situations, human intentions, and causality.
--->The Problem
The task uses the Visual Commonsense Reasoning (VCR) dataset. For every image and question, the AI must do two things:

Stage 1 (Q→A): Pick the correct answer from 4 options.

Stage 2 (QA→R): Pick the correct rationale from 4 options, explaining why it chose that answer.

The final goal is to get a high Q→AR Accuracy (getting both the answer and the rationale correct).

--->Results
We tested the model on 50 random questions from the validation set. Since random guessing is 25%, our results are very strong for a zero-shot model:

Q→A Accuracy (Answer): 68.00%,

QA→R Accuracy (Rationale): 56.00%,

Q→AR Accuracy (Both correct): 42.00%

<img width="1812" height="733" alt="image" src="https://github.com/user-attachments/assets/1217d035-5ee4-445d-946c-f1417ca1d0d3" />

--->Our Approach (Simple Breakdown)
1. The Model
We used Qwen2.5-VL-7B-Instruct, a powerful open-source Vision-Language Model (VLM).

Vision Part: It looks at the image.
Language Part: It reads the question and writes the answer.
Memory Optimization: Because this model is huge (7 Billion parameters), we loaded it using 4-bit quantization. This shrinks the model size so it can run on a standard free Kaggle GPU without crashing.

2. Visual Grounding (Drawing Boxes)
The VCR dataset provides coordinates for people and objects in the image. Instead of giving the AI raw numbers, we used the Python imaging library (PIL) to draw colored bounding boxes with labels (like [person0], [person1]) directly onto the images.

Why? Because the questions reference these labels. If the question asks "Why is [person0] laughing?", the AI can literally look at the image, find the box labeled [person0], and analyze their face.

3. Smart Prompting (Zero-Shot)
   
We didn't train or fine-tune the model. We just asked it really good questions (Prompt Engineering).

We told the AI to act like an expert in social situations.

We asked it to think step-by-step and give a brief reason.

We forced it to end its answer with a strict format: Answer: <letter> so our code could automatically read which option it chose.


4. Data Streaming

The VCR dataset is 32GB. Kaggle only gives 20GB of disk space. To solve this, we used Hugging Face's streaming=True feature. This allowed us to pull images one by one over the internet without ever downloading the whole dataset to the hard drive.

📁 Project Files

->vcr_data.py - Loads the dataset, draws the bounding boxes on images, and prepares the text.

->prompts.py - Creates the text instructions given to the AI and extracts the chosen letter (A, B, C, D).

->model_utils.py - Loads the Qwen2.5-VL model onto the GPU and generates responses.

->run_eval.py - The main script that runs the whole pipeline and calculates the final accuracy.

kaggle_notebook.ipynb - The interactive notebook where the testing was performed.

--->How to Run:

1.Clone this repository.

2.Install requirements: pip install -r requirements.txt

3.Run the evaluation script:

python run_eval.py --split validation --n_questions 50 --out predictions.jsonl

(Note: You need a GPU with at least 16GB VRAM for this to run efficiently).

