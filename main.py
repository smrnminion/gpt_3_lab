import streamlit as st_app
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

gen_config = {
    "max_length": 250,
    "temperature": 1.1,
    "top_p": 2,
    "num_beams": 10,
    "repetition_penalty": 1.5,
    "num_return_sequences": 9,
    "no_repeat_ngram_size": 2,
    "do_sample": True
}

model_identifier = 'ai-forever/rugpt3small_based_on_gpt2'
gpt_model = GPT2LMHeadModel.from_pretrained(model_identifier).to('cpu')
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_identifier)

text_gen_pipeline = pipeline('text-generation', model=gpt_model,
                             tokenizer=gpt_tokenizer, device=-1)

def create_text(input_text):
    output_text = text_gen_pipeline(input_text, **gen_config)[0]['generated_text']
    return output_text

st_app.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st_app.title('Инструмент для генерации аннотаций и введений для научно-исследовательской работы')

user_input = st_app.text_area('Введите ваш текст здесь:', '', height=200, key='input_key')

generate_btn = st_app.button("Запустить генерацию")

if generate_btn: 
    with st_app.spinner("Генерация в процессе..."):
        result_text = create_text(user_input)
        st_app.write(result_text)
