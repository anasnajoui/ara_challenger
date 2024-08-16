import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
import plotly.colors as colors
import streamlit.components.v1 as components
import plotly.express as px

# Load environment variables
load_dotenv()

class OpenAIClient:
    """Handles interactions with the OpenAI API."""

    def __init__(self):
        """Initialize the OpenAI client with the API key from environment variables."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def challenge_score(self, category, kri, description, risk_level, risk_score):
        """
        Generate a challenge to the risk score using OpenAI's GPT model.

        Args:
            category (str): The risk category.
            kri (str): The Key Risk Indicator.
            description (str): Description of the KRI.
            risk_level (str): The assessed risk level.
            risk_score (float): The assigned risk score.

        Returns:
            str: The generated challenge response.
        """
        prompt = f"""
        Category: {category}
        KRI: {kri}
        Description: {description}
        Risk Level: {risk_level}
        Risk Score: {risk_score}

        Based on the information provided only output the following : 
        Provide one sentence brief concise bullet point of why the score might be too high or too low.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a risk assessment expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    def chatbot_response(self, user_input, context):
        """
        Generate a chatbot response based on user input and context.

        Args:
            user_input (str): The user's question or input.
            context (str): The context information for the chatbot.

        Returns:
            str: The generated chatbot response.
        """
        prompt = f"""
        Context: {context}

        User: {user_input}

        Assistant: As a risk assessment expert, I'll answer your question based on the provided context. 
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a risk assessment expert assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

class ARAParser:
    """Parses Asset Risk Assessment data from an Excel file."""

    RISK_CATEGORIES = ["Counterparty", "Liquidity", "Specific", "Market", "Operational", "ESG", "Phase"]


    @staticmethod
    def load_excel_data(file_path):
        """
        Load data from an Excel file.

        Args:
            file_path: Path to the Excel file.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        return pd.read_excel(file_path, sheet_name="Asset Risk Assessment", engine="openpyxl", header=None)

    @classmethod
    def parse_ara_data(cls, df):

        """
        Parse the Asset Risk Assessment data from the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the ARA data.

        Returns:
            tuple: Containing company_name, assessment_date, total_risk_score, category_data, kri_data
        """

        # Extract basic information from the first row
        company_name = df.iloc[0, 0]
        assessment_date = df.iloc[0, 1]
        total_risk_score = float(df.iloc[0, 3])

        # Initialize variables to track current category and KRI
        current_category = None
        current_kri = None
        
        # Initialize dictionaries to store category and KRI data
        category_data = {}
        kri_data = {}

        # Iterate through each row of the DataFrame, starting from the second row
        for index, row in df.iloc[1:].iterrows():
            print(row)
            print()

            # Skip rows where all values are NaN
            if pd.isna(row).all():
                continue

            # Identify risk category
            if isinstance(row.iloc[0], str) and len(row.iloc[0].split()) == 2 and 'risk' in row.iloc[0].lower():
                category_words = row.iloc[0].lower().split()
                other_word = next(word for word in category_words if word != 'risk')
                if other_word in [cat.lower() for cat in cls.RISK_CATEGORIES]:
                    current_category = row.iloc[0]
                    print(f"Identified category: {current_category}")
                    category_data[current_category] = [(index, row.copy())]  # Add the category row itself
                    print(f"Added category row: {row}")
            
            # Identify KRI
            elif not pd.isna(row.iloc[0]) and not pd.isna(row.iloc[3]):
                if current_category is None:
                    print(f"Warning: Found KRI before any category: {row.iloc[0]}")
                    continue
                
                current_kri = row.iloc[0]
                print(f"Identified KRI: {current_kri}")
                
                category_data[current_category].append((index, row))
                kri_data[current_kri] = []

            # Identify Sub-KRI
            elif not pd.isna(row.iloc[1]) and pd.isna(row.iloc[3]) and not pd.isna(row.iloc[2]):
                if current_kri:
                    kri_data[current_kri].append((index, row))
                    print(f"Identified Sub-KRI for {current_kri}: {row.iloc[1]}")
                else:
                    print(f"Warning: Found Sub-KRI before any KRI: {row.iloc[1]}")

        print("\nParsing Results:")
        print(f"Categories found: {list(category_data.keys())}")
        print(f"Categories with data: {[cat for cat, data in category_data.items() if data]}")
        print(f"Total KRIs found: {len(kri_data)}")

        return company_name, assessment_date, total_risk_score, category_data, kri_data

class StreamlitApp:
    """Manages the Streamlit application interface."""

    def __init__(self):
        """Initialize the Streamlit app with OpenAI client and ARA parser."""
        self.openai_client = OpenAIClient()
        self.ara_parser = ARAParser()
        self.df = None  # Initialize df as None

    def run(self):
        """Run the Streamlit application."""
        # Initialize session state variables
        if "risk_level_changes" not in st.session_state:
            st.session_state.risk_level_changes = []
        if "challenges" not in st.session_state:
            st.session_state.challenges = {}

        st.title("Hi! I'm your Asset Risk Assessment AI 🤖")

        uploaded_file = st.file_uploader("Upload your ARA file", type="xlsm")
        
        if uploaded_file is not None:
            try:
                self.df = self.ara_parser.load_excel_data(uploaded_file)
                company_name, assessment_date, total_risk_score, category_data, kri_data = self.ara_parser.parse_ara_data(self.df)

                if not category_data:
                    st.error("No risk categories found in the uploaded file. Please check the file format.")
                    return

                self.display_summary(company_name, assessment_date, total_risk_score, category_data)
                self.display_risk_details(category_data, kri_data)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")

        # Add JavaScript for programmatic tab switching
        st.markdown("""
        <script>
        const urlParams = new URLSearchParams(window.location.search);
        const activeTab = urlParams.get('active_tab');
        if (activeTab) {
            const tabButton = document.querySelector(`button[data-baseweb="tab"] span:contains("${activeTab}")`);
            if (tabButton) {
                tabButton.click();
            }
        }
        </script>
        """, unsafe_allow_html=True)

    def display_summary(self, company_name, assessment_date, total_risk_score, category_data):
        """Display the summary of risk scores."""
        st.title(f"{company_name} - {assessment_date}")

        with st.container():
            st.subheader("Risk Score Summary")
            summary_data = []
            
            for category, rows in category_data.items():
                if rows:
                    category_row = rows[0][1]
                    category_score = float(category_row.iloc[3])
                    
                    if 0 < category_score < 1:
                        category_score = category_score * total_risk_score
                    
                    summary_data.append({"Category": category, "Risk Score": category_score})
                else:
                    summary_data.append({"Category": category, "Risk Score": 0.0})

            summary_df = pd.DataFrame(summary_data).sort_values("Risk Score", ascending=True)
            
            # Create a bar chart using Plotly Express with green-red color scale
            fig = px.bar(summary_df, 
                         x='Risk Score', 
                         y='Category', 
                         orientation='h',
                         text='Risk Score',
                         color='Risk Score',
                         color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green reversed
                         range_color=[0, 10])  # Assuming max score is 10
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                title='Risk Score Summary',
                xaxis_title='Risk Score',
                yaxis_title='Category',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(title="Risk Score"),
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Create a gauge chart for total risk score
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = total_risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Total Risk Score"},
                gauge = {
                    'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 2], 'color': 'green'},
                        {'range': [2, 4], 'color': 'lightgreen'},
                        {'range': [4, 6], 'color': 'yellow'},
                        {'range': [6, 8], 'color': 'orange'},
                        {'range': [8, 10], 'color': 'red'}
                    ],
                }
            ))
            
            gauge_fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=30, b=0),
            )

            st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.markdown("---")

    def display_risk_details(self, category_data, kri_data):
        sorted_categories = sorted([(category, float(rows[0][1].iloc[3])) for category, rows in category_data.items() if rows],
                                   key=lambda x: x[1], reverse=True)
        tab_titles = [category for category, _ in sorted_categories]
        
        # Add an anchor for the category selector
        st.markdown('<a name="category-selector"></a>', unsafe_allow_html=True)
        st.markdown("## Category Selector")
        
        tabs = st.tabs(tab_titles)

        for i, (tab, (category, category_score)) in enumerate(zip(tabs, sorted_categories)):
            with tab:
                st.metric(label="Category Risk Score", value=f"{category_score:.2f}")

                for index, row in category_data[category][1:]:  # Skip the category row
                    kri_name = row.iloc[0]
                    kri_description = row.iloc[1]
                    risk_level = row.iloc[2]
                    risk_score = float(row.iloc[3])

                    with st.expander(label=f"**{kri_name} - {risk_level}**", expanded=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(kri_description)
                            
                            # Display sub-KRIs if they exist
                            if kri_name in kri_data and kri_data[kri_name]:
                                st.write("Sub-KRIs:")
                                for sub_index, sub_kri in kri_data[kri_name]:
                                    with st.container(border=True):
                                        sub_kri_title = sub_kri.iloc[0]  # Title of sub-KRI
                                        sub_kri_description = sub_kri.iloc[1]   # Description of sub-KRI
                                        sub_kri_risk_level = sub_kri.iloc[2]  # Risk level of sub-KRI
                                        st.markdown(f"**{sub_kri_title}** : {sub_kri_description}")
                                        st.markdown(self.get_colored_risk_level(sub_kri_risk_level), unsafe_allow_html=True)

                        with col2:
                            # Create a gauge chart for the risk score
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = risk_score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                gauge = {
                                    'axis': {'range': [None, 10], 'tickwidth': 1},
                                    'bar': {'color': "darkblue"},
                                    'steps' : [
                                        {'range': [0, 2], 'color': "lightgreen"},
                                        {'range': [2, 4], 'color': "yellow"},
                                        {'range': [4, 6], 'color': "orange"},
                                        {'range': [6, 8], 'color': "salmon"},
                                        {'range': [8, 10], 'color': "red"}],
                                }
                            ))
                            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)

                        col3, col4 = st.columns(2)
                        with col3:
                            if st.button(f"Challenge Score", key=f"challenge_{category}_{kri_name}"):
                                challenge = self.openai_client.challenge_score(category, kri_name, kri_description, risk_level, risk_score)
                                st.session_state.challenges[f"challenge_{category}_{kri_name}"] = challenge
                                st.write("Challenge:", challenge)
                        
                        with col4:
                            risk_levels = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
                            new_risk_level = st.selectbox(f"Change Risk Level", 
                                                          options=risk_levels,
                                                          index=risk_levels.index(risk_level),
                                                          key=f"risk_level_{category}_{kri_name}")
                            
                            if new_risk_level != risk_level:
                                comment = st.text_area("Explain the reason for change:", key=f"comment_{category}_{kri_name}")
                                if st.button("Confirm Change", key=f"confirm_{category}_{kri_name}"):
                                    if comment:
                                        st.session_state.risk_level_changes.append({
                                            "category": category,
                                            "kri": kri_name,
                                            "old_level": risk_level,
                                            "new_level": new_risk_level,
                                            "comment": comment
                                        })
                                        st.success(f"Risk level for {kri_name} changed from {risk_level} to {new_risk_level}")
                                    else:
                                        st.warning("Please provide a comment explaining the change.")

                # Add "Go to Top" link
                st.markdown('''
                    <a href="#category-selector">
                        <button style="
                            background-color: #4CAF50;
                            border: none;
                            color: white;
                            padding: 15px 32px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin: 4px 2px;
                            cursor: pointer;">
                            Back to Top
                        </button>
                    </a>
                ''', unsafe_allow_html=True)

        # Display risk level changes in the sidebar
        with st.sidebar:
            st.header("Risk Level Changes")
            
            # Create a scrollable container
            changes_container = st.container()
            with changes_container:
                changes_text = ""
                
                # Add risk level changes
                if st.session_state.risk_level_changes:
                    for change in st.session_state.risk_level_changes:
                        changes_text += f"{change['category']} : {change['kri']}\n"
                        changes_text += f"{change['old_level']} → {change['new_level']}\n"
                        changes_text += f"Reason: {change['comment']}\n\n"
                
                if changes_text:
                    st.code(changes_text, language="text")
                else:
                    st.write("No risk level changes yet.")

            # Add delete last entry button
            if st.session_state.risk_level_changes:
                if st.button("Delete Last Change"):
                    st.session_state.risk_level_changes.pop()
                    st.experimental_rerun()

            # Add chatbot to the sidebar
            st.markdown("---")
            st.header("Ask me anything...💬")
            self.display_chatbot()

    def display_chatbot(self):
        """Display the chatbot interface in the sidebar."""
        if self.df is not None:
            context = "\n".join([f"Category: {row.iloc[0]}, KRI: {row.iloc[1]}, Risk Level: {row.iloc[2]}, Risk Score: {row.iloc[3]}" 
                                 for _, row in self.df.iterrows() if not pd.isna(row.iloc[1])])

            if "messages" not in st.session_state:
                st.session_state.messages = []

            messages = st.container(height=200)  # Fixed height of 200 pixels
            for message in st.session_state.messages:
                messages.chat_message(message["role"]).write(message["content"])

            if prompt := st.chat_input("Ask about the risk assessment:"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                messages.chat_message("user").write(prompt)

                response = self.openai_client.chatbot_response(prompt, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                messages.chat_message("assistant").write(response)
        else:
            st.write("Please upload an ARA file to use the chat feature.")

    def get_colored_risk_level(self, risk_level):
        color = {
            "Low risk": "green",
            "Medium-low risk": "lightgreen",
            "Medium risk": "yellow",
            "Medium-high risk": "orange",
            "High risk": "red"
        }.get(risk_level, "black")
        return f'<span style="color: {color};"><strong>{risk_level}</strong></span>'

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()