import os
from pathlib import Path
import gradio as gr
from CreditCard.entity import CreditData
import numpy as np
import pandas as pd 
from CreditCard.utils import load_object

def predict_credit_card_fraud(LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_O, PAY_1, PAY_2,
                              PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5,
                              BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, metadata):
    try:
        credit_data = CreditData(LIMIT_BAL=LIMIT_BAL, SEX=SEX, EDUCATION=EDUCATION, MARRIAGE=MARRIAGE, AGE=AGE,
                                PAY_0=PAY_O, PAY_2=PAY_2, PAY_3=PAY_3, PAY_4=PAY_4, PAY_5=PAY_5, PAY_6=PAY_6,
                                BILL_AMT1=BILL_AMT1, BILL_AMT2=BILL_AMT2, BILL_AMT3=BILL_AMT3, BILL_AMT4=BILL_AMT4,
                                BILL_AMT5=BILL_AMT5, BILL_AMT6=BILL_AMT6, PAY_AMT1=PAY_AMT1, PAY_AMT2=PAY_AMT2,
                                PAY_AMT3=PAY_AMT3, PAY_AMT4=PAY_AMT4, PAY_AMT5=PAY_AMT5, PAY_AMT6=PAY_AMT6)
        DATA = credit_data.dict()

        DATA_TO_PREDICT = pd.DataFrame(DATA, index=[0])
        model = load_object(Path('production_model/best_model.pkl'))
        predict_binary = [1]
        return predict_binary[0]
    except Exception as e:
        raise e



with gr.Blocks() as demo:
    gr.Markdown("## Credit Card Fraud Detection")
    with gr.Row():
        with gr.Column():
            LIMIT_BAL = gr.inputs.Slider(minimum=5000, maximum=1000000, default=100000, label="Amount of the given credit (NT dollar)")
            SEX =gr.inputs.Dropdown(choices=[("Male", 1), ("Female", 2)], label="Gender", default=1)
            EDUCATION = gr.inputs.Dropdown(choices=[("Graduate School", 1), ("University", 2), ("High School", 3), ("Others", 4)], label="Education", default=1)
            MARRIAGE = gr.inputs.Dropdown(choices=[("Married", 1), ("Single", 2), ("Others", 3)], label="Marriage", default=1)
            AGE = gr.inputs.Slider(minimum=20, maximum=100, default=30, label="Age in years")
            PAY_O = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in September 2005 (scale same as above)")
            PAY_1 = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in August 2005 (scale same as above)")
            PAY_2 = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in July 2005 (scale same as above)")
            PAY_3 = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in June 2005 (scale same as above)")
            PAY_4 = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in May 2005 (scale same as above)")
            PAY_5 = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in April 2005 (scale same as above)")
            PAY_6 = gr.inputs.Slider(minimum=0, maximum=10, default=0, label="Repayment status in March 2005 (scale same as above)")
            BILL_AMT1 = gr.inputs.Slider(minimum=-35000, maximum=1000000, default=0, label="Amount of bill statement in September 2005 (NT dollar)")
            BILL_AMT2 = gr.inputs.Slider(minimum=-35000, maximum=1000000, default=0, label="Amount of bill statement in August 2005 (NT dollar)")
            BILL_AMT3 = gr.inputs.Slider(minimum=-35000, maximum=1000000, default=0, label="Amount of bill statement in July 2005 (NT dollar)")
            BILL_AMT4 = gr.inputs.Slider(minimum=-35000, maximum=1000000, default=0, label="Amount of bill statement in June 2005 (NT dollar)")
            BILL_AMT5 = gr.inputs.Slider(minimum=-35000, maximum=1000000, default=0, label="Amount of bill statement in May 2005 (NT dollar)")
            BILL_AMT6 = gr.inputs.Slider(minimum=-35000, maximum=1000000, default=0, label="Amount of bill statement in April 2005 (NT dollar)")
            PAY_AMT1 = gr.inputs.Slider(minimum=0, maximum= 2000000, default=0, label="Amount of previous payment in September 2005 (NT dollar)")
            PAY_AMT2 = gr.inputs.Slider(minimum=0, maximum= 2000000, default=0, label="Amount of previous payment in August 2005 (NT dollar)")
            PAY_AMT3 = gr.inputs.Slider(minimum=0, maximum= 2000000, default=0, label="Amount of previous payment in July 2005 (NT dollar)")
            PAY_AMT4 = gr.inputs.Slider(minimum=0, maximum= 2000000, default=0, label="Amount of previous payment in June 2005 (NT dollar)")
            PAY_AMT5 = gr.inputs.Slider(minimum=0, maximum= 2000000, default=0, label="Amount of previous payment in May 2005 (NT dollar)")
            PAY_AMT6 = gr.inputs.Slider(minimum=0, maximum= 2000000, default=0, label="Amount of previous payment in April 2005 (NT dollar)")
            
        with gr.Column():
            label = gr.Label("Prediction")
            with gr.Row():
                predict_btn = gr.Button(value="Predict")
            predict_btn.click( predict_credit_card_fraud,
                              inputs=[LIMIT_BAL , SEX , EDUCATION , MARRIAGE , AGE , PAY_O , PAY_1 , PAY_2 ,
                                      PAY_3 , PAY_4 , PAY_5 , PAY_6 , BILL_AMT1 , BILL_AMT2 , BILL_AMT3 , BILL_AMT4 , BILL_AMT5 , 
                                      BILL_AMT6 , PAY_AMT1 , PAY_AMT2 , PAY_AMT3 , PAY_AMT4 , PAY_AMT5 , PAY_AMT6],outputs=[label])

demo.launch()




 

