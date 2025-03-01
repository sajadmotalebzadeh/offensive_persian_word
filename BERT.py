import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import json

# ------------------------------
# Load Model, Tokenizer, and Threshold (cached for efficiency)
# ------------------------------
@st.cache_resource
def load_model_tokenizer_and_threshold():
    model = TFAutoModelForSequenceClassification.from_pretrained("sajadmotalebzadeh/my-finetuned-parsbert")
    tokenizer = AutoTokenizer.from_pretrained("saved_tokenizer")
    try:
        with open("threshold.json", "r") as f:
            threshold = data.get("threshold", 0.5)
    except Exception as e:
        threshold = 0.5
    return model, tokenizer, threshold

model, tokenizer, optimal_threshold = load_model_tokenizer_and_threshold()

st.title("Persian Offensive Word Detection")
st.write("This tool detects offensive language in Persian text using a fine-tuned ParsBERT model.")

# ------------------------------
# Persian Text Normalization & Preprocessing
# ------------------------------
def normalize_persian(text):
    # Replace common variations in Persian characters
    return text.replace("ي", "ی").replace("ك", "ک")

def preprocess_text(text, tokenizer, max_length=128):
    text = normalize_persian(text)
    encoding = tokenizer(text, return_tensors="tf", padding="max_length", truncation=True, max_length=max_length)
    return encoding

# ------------------------------
# Offensive Words Extraction (simple lexicon-based check)
# ------------------------------
def get_offensive_words(text):
    known_offensive_words = {
        "قرومساق", "کله تخم مرغی", "کیری", "لعنتی", "احمق", "کثافت", "چس", "گوز", "ان", "لجن",
        "بی شرف", "بیشعور", "گوه", "کون", "کسکش", "سگ پدر", "پدرسگ", "شاش", "ریدن", "ریدی",
        "دیوس", "انی", "گهی", "بی پدر", "مادرسگ", "جنده", "گایدی", "گایدن", "کیر", "عمتو",
        "خفه شو", "خفه", "خفه خون", "مرض داری", "گردن دراز", "خری", "گاوی", "اسبی", "سگی",
        "حیوانی", "دهنتوببند", "انگل", "آشغال", "خرفت", "پپه", "خنگ", "دکل", "دله", "قرتی",
        "گوزو", "کونده", "کون ده", "گاگول", "ابله", "گنده گوز", "کس", "خارکیونی", "کله کاندومی",
        "گشاد", "دخترقرتی", "خواهرجنده", "مادرجنده", "لخت", "بخورش", "بپرسرش", "بپرروش",
        "بیابخورش", "میخوریش", "بمال", "دیوس خان", "زرنزن", "زنشو", "زنتو", "زن جنده",
        "بکنمت", "بکن", "بکن توش", "بکنش", "لز", "سکس", "سکسی", "ساک", "ساک بزن", "پورن",
        "سکسیی", "کونن", "کیرر", "بدبخت", "خایه", "خایه مال", "خایه خور", "ممه", "ممه خور",
        "دخترجنده", "کس ننت", "کیردوس", "مادرکونی", "خارکسده", "خارکس ده", "کیروکس", "کس و کیر",
        "زنا", "زنازاده", "ولدزنا", "ملنگ", "سادیسمی", "فاحشه", "خانم جنده", "فاحشه خانم",
        "سیکتیر", "سسکی", "کس خیس", "حشری", "گاییدن", "بکارت", "داف", "بچه کونی", "کسشعر",
        "سرکیر", "سوراخ کون", "حشری شدن", "کس کردن", "کس دادن", "بکن بکن", "شق کردن",
        "کس لیسیدن", "آب کیر", "جاکش", "جلق زدن", "جنده خانه", "شهوتی", "عن", "قس", "کردن",
        "کردنی", "کس کش", "کوس", "کیرمکیدن", "لاکونی", "پستان", "پستون", "آلت", "آلت تناسلی",
        "نرکده", "مالوندن", "سولاخ", "جنسی", "دوجنسه", "سگ تو روحت", "بی غیرت", "نعشه", "بی عفت",
        "مادرقهوه", "پلشت", "پریود", "کله کیری", "کیرناز", "پشمام", "لختی", "کسکیر", "دوست دختر",
        "دوست پسر", "کونشو", "دول", "شنگول", "کیردراز", "داف ناز", "سکسیم", "کوص", "ساکونی",
        "کون گنده", "سکسی باش", "کسخل", "کصخل", "کصکلک بازی", "صیغه ای", "گوش دراز", "درازگوش",
        "خز", "ماچ", "ماچ کردنی", "اسکل", "هیز", "بیناموس", "بی آبرو", "لاشی", "لاش گوشت",
        "باسن", "جکس", "سگ صفت", "کصکش", "مشروب", "عرق خور", "سکس چت", "جوون", "سرخور",
        "کلفت", "حشر", "لاس", "زارت", "رشتی", "ترک", "فارس", "لر", "عرب", "خر", "گاو",
        "اسب", "گوسفند", "کرم", "الاق", "الاغ", "احمق", "بی شعور", "حرومزاده", "کونی", "گه",
        "مادر جنده", "کث", "کص", "پسون", "خارکسّه", "دهن گاییده", "دهن سرویس", "پدر سگ",
        "پدر سوخته", "پدر صلواتی", "لامصب", "زنیکه", "مرتیکه", "مردیکه", "بی خایه", "عوضی",
        "اسگل", "اوسکل", "اوسگل", "اوصگل", "اوصکل", "دیوث", "دیوص", "قرمصاق", "قرمساق",
        "غرمساق", "غرمصاق", "فیلم سوپر", "چاقال", "چاغال", "چس خور", "کس خور", "کس خل",
        "کوس خور", "کوس خل", "کص لیس", "کث لیس", "کس لیس", "کوص لیس", "کوث لیس", "کوس لیس",
        "اوبی", "خارکونی", "کونی مقام", "شاش خالی", "دلقک", "عن دونی", "خار سولاخی", "سولاخ مادر",
        "عمه ننه", "خارتو", "بو زنا", "شاش بند", "کیونی", "کصپدر", "شغال", "خپل", "ساکر",
        "زن قوه", "پشم کون", "جنده پولی", "حرومی", "دودول طلا", "چوسو", "هزار پدر", "بی فانوس",
        "پرده زن", "آبم اومد", "چس خوری", "زاخار", "گی مادر", "ظنا", "بی پدرو مادر", "کیرم دهنت",
        "بکیرم", "به تخم اقام", "کیر خر", "ننه مرده", "سلیطه", "لاشخور", "هرزه", "حروم\u200cلقمه",
        "پاچه\u200cخوار", "ارگاسم", "دول ننه", "مادر فاکر", "کصپولی", "ننه هزار کیر", "قرمدنگ",
        "توله سگ", "جفنگ", "ریدم", "شومبول", "دهنتو گاییدم", "چسو", "بی عرضه", "بی مصرف",
        "بدطینت", "خبیث", "زالو", "مغز پریودی", "کسپولی", "ناکس", "مفت\u200cخور", "چرب کنش",
        "اوب", "فرو کن", "بچه کیونی"
    }
    # A simple approach: split on whitespace and check each word
    words = text.split()
    offensive_detected = [word for word in words if word in known_offensive_words]
    return list(set(offensive_detected))

# ------------------------------
# Streamlit Interface
# ------------------------------
user_input = st.text_area("Enter Persian text:")

if st.button("Analyze"):
    if user_input.strip():
        # Preprocess the input text
        encoding = preprocess_text(user_input, tokenizer)
        outputs = model(encoding)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        offensive_prob = probabilities[1]  # probability for offensive class
        
        # Use the tuned threshold from training
        result = "Offensive" if offensive_prob > optimal_threshold else "Non-Offensive"
        offensive_words = get_offensive_words(user_input)
        
        st.subheader("Result:")
        st.write(f"Prediction: **{result}** (Offensive probability: {offensive_prob:.2f}, threshold: {optimal_threshold:.2f})")
        if result == "Offensive" and offensive_words:
            st.markdown("### Offensive Words Detected:")
            st.write(", ".join(offensive_words))
        elif result == "Offensive":
            st.write("No specific offensive words were identified.")
    else:
        st.warning("Please enter some text for analysis.")

st.write("---")
