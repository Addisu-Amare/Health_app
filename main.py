#import streamlit library
import streamlit as st

st.set_page_config(
    page_title="HealthCare Hub - Your Complete Wellness Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            background-color: #0d1b2a;
            color: #e0e0e0;
        }

        /* Main Title */
        .title {
            font-size: 50px;
            font-weight: 600;
            color: #1e90ff;
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 0px 0px 10px rgba(30, 144, 255, 0.5);
        }

        /* Subtitle - Made More Visible */
        .subtitle {
            font-size: 28px;
            color: #ffffff !important;
            text-align: center;
            font-weight: 400;
            margin-bottom: 40px;
            background: linear-gradient(90deg, transparent, #333333, transparent);
            padding: 15px;
            border-radius: 10px;
            text-shadow: 2px 2px 4px #000000;
        }
        
        .subtitle-small {
            font-size: 20px;
            color: #ffd700 !important;
            text-align: center;
            font-weight: 300;
            margin-top: -20px;
            margin-bottom: 30px;
            background-color: rgba(0,0,0,0.5);
            padding: 8px;
            border-radius: 5px;
        }

        /* Sections with Red Headings */
        .section {
            background: #1e1e1e;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
            transition: all 0.3s ease-in-out;
        }

        .section:hover {
            transform: translateY(-5px);
            box-shadow: 0px 6px 16px rgba(255, 255, 255, 0.2);
        }

        .red-heading {
            color: #ff3333 !important;
            font-weight: 600;
            border-bottom: 2px solid #ff3333;
            padding-bottom: 8px;
            margin-bottom: 15px;
            font-size: 24px;
        }

        /* Sidebar */
        .sidebar-text {
            font-size: 18px;
            color: #ffffff;
            text-align: center;
            margin-bottom: 20px;
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 8px;
        }

        /* Contact Section */
        .contact {
            text-align: left;
            font-size: 20px;
            color: #ffffff;
            margin-top: 40px;
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 15px;
        }

        .contact a {
            color: #1e90ff;
            text-decoration: none;
            font-weight: bold;
        }
        
        .contact a:hover {
            text-decoration: underline;
        }

        /* Profile Links */
        .profile-links {
            text-align: center;
            margin-top: 20px;
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 15px;
        }

        .profile-links a {
            font-size: 20px;
            color: #1e90ff;
            text-decoration: none;
            margin: 0 15px;
            font-weight: 600;
        }
        
        .profile-links a:hover {
            color: #ff3333;
            text-decoration: underline;
        }
        
        /* Feature Description Text */
        .feature-text {
            color: #e0e0e0;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .amharic-text {
            color: #ffd700;
            font-weight: 500;
        }
        
        .english-text {
            color: #b0e0e6;
            font-weight: 400;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("<h2 style='color: #ff3333; border-bottom: 2px solid #ff3333; padding-bottom: 8px;'>📌 አሰሳ / NAVIGATION</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class='sidebar-text'>
    <span class='amharic-text'>የAI ጤና አጠባበቅ አውታረ መረብን ለማሰስ የጎን አሞሌን ይጠቀሙ</span><br>
    <span class='english-text'>Use the sidebar to explore different features</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.image("utils/ph3.jpg", use_container_width=True)

# Main Content
st.markdown("<div class='title'>🩺 HealthCare Hub - አጠቃላይ የደህንነት ረዳትዎ</div>", unsafe_allow_html=True)

# Visible Subheadings
st.markdown("""
<div class='subtitle'>
    <span class='amharic-text'>በአርቴፊሻል ኢንተሊጀንስ በሚመሩ ትንበያዎች እና ግንዛቤዎች የጤና አጠባበቅን መቀየር</span>
</div>
<div class='subtitle-small'>
    <span class='english-text'>Transforming healthcare with AI-driven predictions and insights</span>
</div>
""", unsafe_allow_html=True)

st.image("utils/ph1.jpeg", use_container_width=True)

# Features Section with Red Heading
st.markdown("<h2 style='text-align: center; color: #ff3333; font-size: 36px; margin: 30px 0;'>✨ ባህሪያት / FEATURES</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='section'>
        <h3 class='red-heading'>💡 በሽታ መመርመሪያ / Disease Prediction</h3>
        <p class='feature-text'>
            <span class='amharic-text'>ምልክቶችን በመተንተን በሽታዎችን መተንበይ</span><br>
            <span class='english-text'>Analyze symptoms and predict possible diseases using advanced AI models</span>
        </p>
    </div>

    <div class='section'>
        <h3 class='red-heading'>💊 መድሀኒት ምክር / Drug Recommendation</h3>
        <p class='feature-text'>
            <span class='amharic-text'>በሕክምና ታሪክ መሰረት የመድሀኒት ምክር ማግኘት</span><br>
            <span class='english-text'>Get AI-powered medication suggestions based on medical history and diagnosis</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='section'>
        <h3 class='red-heading'>❤️ የልብ ሕመም ምዘና / Heart Risk Assessment</h3>
        <p class='feature-text'>
            <span class='amharic-text'>የልብ ጤናዎን ይገምግሙ እና ትንበያ ያግኙ</span><br>
            <span class='english-text'>Assess your heart health and receive AI-powered risk analysis</span>
        </p>
    </div>

    <div class='section'>
        <h3 class='red-heading'>🤖 AI ቻትቦት / LLM Chatbot</h3>
        <p class='feature-text'>
            <span class='amharic-text'>ስለጤናዎ ለማወቅ ከAI ረዳት ጋር ይነጋገሩ</span><br>
            <span class='english-text'>Chat with an AI-powered assistant to get health insights and recommendations</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Technologies Used with Red Heading
st.markdown("<h2 style='text-align: center; color: #ff3333; font-size: 36px; margin: 30px 0;'>⚙️ ቴክኖሎጂዎች / TECHNOLOGIES USED</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='section'>
        <h3 class='red-heading'>🤖 ማሽን ለርኒንግ / Machine Learning</h3>
        <p class='feature-text'>
            <span class='amharic-text'>RandomForest, XGBoost እና ጥልቅ ትምህርት ሞዴሎችን መጠቀም</span><br>
            <span class='english-text'>Utilizing RandomForest, XGBoost, and Deep Learning models</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='section'>
        <h3 class='red-heading'>🗂 NLP እና AI / NLP & AI</h3>
        <p class='feature-text'>
            <span class='amharic-text'>ለቻትቦት መስተጋብር የተፈጥሮ ቋንቋ ማቀናበሪያ መጠቀም</span><br>
            <span class='english-text'>Leveraging Natural Language Processing for chatbot interactions</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='section'>
        <h3 class='red-heading'>☁️ ክላውድ ኮምፒውቲንግ / Cloud Computing</h3>
        <p class='feature-text'>
            <span class='amharic-text'>AWS, GCP እና Streamlit Cloud ላይ መተግበር</span><br>
            <span class='english-text'>Deployed using AWS, GCP, and Streamlit Cloud</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Why Use This App with Red Heading
st.markdown('''
<div style="background:#111; padding:15px; border:2px solid red; border-radius:10px">
    <h3 style="color:red; text-align:center">🔍 WHY USE THIS?</h3>
    <p><span style="color:red">✅</span> <span style="color:gold">ትክክለኛ ትንበያ</span><br><span style="color:#ccc">Accurate Predictions</span></p>
    <p><span style="color:red">✅</span> <span style="color:gold">ፈጣን እገዛ</span><br><span style="color:#ccc">Real-Time Assistance</span></p>
    <p><span style="color:red">✅</span> <span style="color:gold">ለተጠቃሚ ምቹ</span><br><span style="color:#ccc">User-Friendly</span></p>
    <p><span style="color:red">✅</span> <span style="color:gold">አስተማማኝ</span><br><span style="color:#ccc">Secure & Reliable</span></p>
</div>
''', unsafe_allow_html=True)


st.markdown("---")

# Contact Us with Red Heading (Left Aligned)
st.markdown("""
<div class='contact'>
    <h2 style='color: #ff3333; font-size: 32px; margin-bottom: 20px; border-bottom: 2px solid #ff3333; padding-bottom: 8px;'>📬 ያግኙን / CONTACT US</h2>
    <p class='feature-text'>
        <span class='amharic-text' style='font-size:18px;'>ጥያቄዎች አሉዎት? እባክዎ እዚህ ያግኙን</span><br>
        <span class='english-text' style='font-size:16px;'>Have questions? Reach out to us at:</span>
    </p>
    <p style='font-size:20px; margin-top:15px;'>📧 <a href="mailto:0941813057estifanos@gmail.com">0941813057estifanos@gmail.com</a></p>
</div>
    """, unsafe_allow_html=True)

# Profile Links with Red Heading (Centered)
st.markdown("""
<div class='profile-links'>
    <h2 style='color: #ff3333; font-size: 32px; margin-bottom: 20px;'>🌐 ከእኔ ጋር ይገናኙ / CONNECT WITH ME</h2>
    <p style='font-size:24px; margin:20px 0;'>
        <a href="https://github.com/Addisu-Amare" target="_blank">GitHub</a> | 
        <a href="www.linkedin.com/in/addisu-amare-2643ba16a" target="_blank">LinkedIn</a>
    </p>
    <p style="font-size: 18px; color: #ffd700; margin-top: 10px; background-color: #2d2d2d; padding: 10px; border-radius: 8px;">
        🇪🇹 በማህበራዊ ትስስር ይከታተሉኝ | 🇬🇧 Follow me on social media
    </p>
</div>
    """, unsafe_allow_html=True)