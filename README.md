# corona_virus_NLP
This project is a final assignment in Advanced topics in Advanced Topics in Deep Learning.

Authors: Itai Peleg, Anat Korol Gordon

In this project, we finetuned and explored Covid-tweet-Bert and BerTweet for covid related tweet classificationclassification of sentiment


# First, put our models to that test with your own tweets
Want to test our model's performance? want to compare fine tuned models with distilled ones? You can do it interactively!
Run our custom website we built using Streamlit package, generate a tweet and see how our models perform.
You can compare performance, as well as inference time

## Before you run the app, make sure to finish setup:
1. Create your own virtual environment
2. nstall requirements - pip install -r requirements.txt
3.  Dowanload state dict folders from the following links:

**results**: [google drive folder] (https://drive.google.com/drive/folders/15cztgT7NYWh6qXfY8NOyUhsl0Speb8R8?usp=sharing)
please place the folder in the project root directory, and make sure the following paths exist:
PROJECT_ROOT / results / best_models (4 distinct models in the format <model_key>_study_augmented_state_dict.py)
PROJECT_ROOT / results / best_compressed (2 distinct models in the format: <model_key>_knowledge_distillation_model.py)

**saved_compressed**: [google drive folder](https://drive.google.com/drive/folders/18gdVRAE-tWaN5vmmYwFD3Y9EeRD6iZgt?usp=sharing)
Please place the folder inside the "compression" folder, so that you have: 
Project_Root / Compression / saved_compressed / 4 sub folders, each with a state_dict and summary file, in the following pattern: <model_key>_knowledge_distillation_state_dict.py,     
<model_key>_knowledge_distillation_summary.json

4. Now, run the website app in the command line: 
**streamlit run app.py**
you will be directed to a locally hosted site (press "open in browser"), and after website loading you will be prompted to Enter tweet text and view the model's predictions, confidence, and also processing time. Feel free to change models in the left bar and explore for your self. Please note that first run will take a few minutes because models are loaded. 
We reccomend to test a tweet and observe the differences in inference time and class distribution between teacher models ( fine tuned Covidbert and Bertweet) and fine tuned students (DistiltBert and DistilRoberta).

# Further evaluation of project:
For evaluating *ALL* of our models and receive a detailed report of their performance, accuracy metrics, size and inference time, go back to your IDE and run the command: 
**python main_compress.py** 
in terminal. This will initialize an evaluation process that will be printed for you in terminal.
### keyword arguments:
model_key: currently runs all models, you can select one
distill_epochs: number of epochs to train distillation if you choose to train
do_train: not training - loads state dicts from the already compressed models 
do_save_models: choose False unless you wish to save 
do_save_reports: False unless you wish to save. Reports can be viewed in "reports" folder
temperature: modify temperature for distillation, default is 3.0
alpha: modify alpha for distillation, default is 0.7
max_samples: max samples for evaluation, small amount for quick testing, we used the whole dataset

# And now, how to navigate in our project?
### EDA - 
To look at the data exploration and visualizations we did, go to: EDA.ipynb
### For Fine Tuning - 
Please check out "models" folder, where you can find training_HF.py - with HuggingFace Trainer, and training_pytorch.py, with a full pytorch implementation. All training was documented using Optuna and Weights and biases. (Note - If you want to run training, please create a .env file and add your wandb key as WANDB_API_KEY=your-api-key)
### To inspect compression techniques -
Go to "compressions" folder, where you will be provided with functions for dynamic quantization, structured and unstructured pruning and also a Fine Tuning pipeline for knowledge distillation. Full reports of the compression process can be found in the reports folder (PROJECT_ROOT / compression / reports).
