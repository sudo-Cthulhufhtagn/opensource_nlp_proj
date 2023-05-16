#%%
import gradio as gr
import time
from transformers import pipeline
import shap
import matplotlib.pyplot as plt
import os
import json

if os.path.exists('.env'):
    from dotenv import load_dotenv
    load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "RandomCatLover/logs_for_demo_nlp2")


#%%
origin_classifier = pipeline("sentiment-analysis", 
#model="my_awesome_model/checkpoint-504",
model="RandomCatLover/scitech_thesis_classifier",
return_all_scores = True)


def check_input_intr(text):
        text_len = text.split()
        if text_len > 160:
            gr.Error("Text length should be less than 160 words")
            
        elif text_len < 100:
            gr.Error("Text length should be at least 100 words")
        return True


def interpretation_function(text):
    #check_input_intr(text)
    explainer = shap.Explainer(origin_classifier)

    shap_values = explainer([text])
    # Dimensions are (batch size, text size, number of classes)
    # Since we care about positive sentiment, use index 1
    scores = list(zip(shap_values.data[0], shap_values.values[0, :, 1]))

    scores_desc = sorted(scores, key=lambda t: t[1])[::-1]

    # Filter out empty string added by shap
    scores_desc = [t for t in scores_desc if t[0] != ""]

    fig_m = plt.figure()

    # Select top 5 words that contribute to positive sentiment
    plt.bar(x=[s[0] for s in scores_desc[:5]],
            height=[s[1] for s in scores_desc[:5]])
    plt.title("Top words contributing to positive sentiment")
    plt.ylabel("Shap Value")
    plt.xlabel("Word")
    return {"original": text, "interpretation": scores}, fig_m

def classifier(text):
    #check_input(text)
    pred = origin_classifier(text)
    return {p["label"]: p["score"] for p in pred[0]}

callback = gr.CSVLogger()

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("""# Abhuman thesis detection demo
        ## **Important note**: if input is too long it may result in error. Optimal input length 150 words.""")

        with gr.Accordion("Video guide", open=True):
            with gr.Row():
                #gr.Markdown("## Text explanations")
                gr.HTML("""<iframe width="560" height="315" src="https://www.youtube.com/embed/bSq4RM1z5fU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>""")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text")
                with gr.Row():
                    classify = gr.Button("Classify")
                    interpret = gr.Button("Interpret")
            with gr.Column():
                label = gr.Label(label="Predicted")
                with gr.Row():
                    like = gr.Button("👍")
                    dislike = gr.Button("👎")
            with gr.Column():
                gr.Progress(track_tqdm=True)
                #with gr.Tabs():
                with gr.TabItem("Display interpretation with plot"):
                    interpretation_plot = gr.Plot()

                with gr.TabItem("Display interpretation with built-in component"):
                    interpretation = gr.components.Interpretation(input_text)
            

    gr.Examples([["The Finite Element Analysis (FEA) is a commonly employed method in engineering and science. It is used to simulate and analyse the behaviour of complex structures and sys-tems under various conditions by breaking down complex problems into a set of smaller and simpler parts called finite elements and analysing them individually using mathemat-ical modelling that considers the physical properties and behaviours of the materials and structures being analysed. FEA allows simulation of physical phenomena such as stress, strain, and deformation, which is the most important for this work. It is also useful to optimize designs in silico and reduce the time and costs required for physical testing via the elimination of less robust solutions. "],
    ["The batch contains at least 10 micropores to eliminate human error in postprocessing. After instructions are loaded to the memory stick and it is plugged into the printer, the printing process begins. The most tedious step of printing on the SLA printer is post-processing. For this purpose, the washing and curing station is used. Due to complex de-signs and internal volumes photopolymer resin is trapped inside due to the surface ten-sion and it is not possible to wash it off automatically in the washing station. For that each pore needs to be washed with the syringe and for that it has a separate secretion hole for the photopolymer resin to leave when the pore is washed with isopropanol. Unfortu-nately, in the beginning it was common the pores were popped during the washing stage because the photopolymer resin is not fully cured and fragile."]], 
    inputs=[input_text])
    classify.click(classifier, input_text, label, queue=True)
    interpret.click(interpretation_function, input_text, [interpretation, interpretation_plot], queue=True)

    hf_writer.setup([input_text, label], "temp")


    like.click(lambda *args: hf_writer.flag([json.dumps(args[1]), {'text': args[0]}], 'like'), [input_text, label], None, preprocess=False)
    dislike.click(lambda *args: hf_writer.flag([json.dumps(args[1])], "dislike"), [input_text, label], None, preprocess=False)
    #like.click(lambda *text: print(text), [input_text, label], None, preprocess=False)



demo.queue(concurrency_count=3)
#demo.launch(auth=("student", "whataday"))
demo.launch()
#demo.launch(share=True)