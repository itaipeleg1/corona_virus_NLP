# corona_virus_NLP
This project is a final assignment in Advanced topics in Advanced Topics in Deep Learning.

Authors: Itai Peleg, Anat Korol Gordon

In this project, we finetuned and explored Covid-tweet-Bert and BerTweet for covid related tweet classificationclassification of sentiment


# First, put our models to that test with your own tweets
Want to test our model's performance? want to compare fine tuned models with distilled ones? You can do it interactively!
Run our custom website we built using Streamlit package, generate a tweet and see how our models perform.
You can compare performance, as well as inference time

## Before you run the app, make sure to finish setup:
*Create your own virtual environment
*install requirements - pip install -r requirements.txt
Dowanload state dict folders from the following links:

"results": [<link>](https://drive.google.com/drive/folders/15cztgT7NYWh6qXfY8NOyUhsl0Speb8R8?usp=sharing) 
please place the folder in the project root directory (please make sure it contains two sub folders - best_models and best_compressed)

"saved_compressed": [<link>](https://drive.google.com/drive/folders/18gdVRAE-tWaN5vmmYwFD3Y9EeRD6iZgt?usp=sharing)
Please place the folder inside the "compression" folder, so that you have: Project Root > Compression > saved_compressed > 4 sub folders, each with 2 state dicts inside

Now, run the website app in the command line: "streamlit run app.py"
you will be directed to a locally hosted site (press "open in browser"), and after website loading you will be prompted to Enter tweet text and view the model's predictions, confidence, and also processing time. Feel free to change models in the left model bar and explore for your self.

# Further evaluation of project:
For evaluating *ALL* of our models and receive a detailed report of their performance, accuracy metrics, size and inference time, go back to enterpreter and run the command: "python main_compress.py" in terminal. This will initialize  an evaluation process that will be printed for you in terminal (you can change the keyword arguemnts as you desire )
# And now, how to navigate in our project?
### EDA - To look at the data exploration and visualizations we did, go to: EDA.ipynb
### For Fine Tuning - please check out "models" folder, where you can find training_HF.py - with HuggingFace Trainer, and training_pytorch.py, with a full pytorch implementation. All training was documented using Optuna and Weights and biases. (Note - If you want to run training, please create a .env file and add your wandb key as WANDB_API_KEY=your-api-key)
### To inspect compression techniques, go to "compressions" folder, where you will be provided with functions for dynamic quantization, structured and unstructured pruning and also a Fine Tuning pipeline for knowledge distillation. Full reports of the compression process can be found in the reports folder.
### 