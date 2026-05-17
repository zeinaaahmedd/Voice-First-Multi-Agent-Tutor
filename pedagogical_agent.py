import requests
import json

class PedagogicalAgent:
    def __init__(self, model_name="qwen2.5:7b", api_url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.api_url = api_url
        
        # Defining the pedagogical persona in the system prompt.
        # This instructs the model to behave like a supportive Egyptian tutor.
        self.system_prompt = (
            "أنت معلم وميسر تعليمي ذكي، صبور، ومشجع تدعى 'بصيرة' (BASIRA). "
            "مهمتك هي تعليم الطلاب الكفيفين وضعاف البصر باستخدام اللهجة المصرية البسيطة والواضحة. "
            "التزم بالقواعد التالية بدقة:\n"
            "1. تحدث بلهجة مصرية فصيحة ومفهومة تناسب الشرح التعليمي.\n"
            "2. أسلوبك يجب أن يكون تفاعلياً، مشجعاً، ومليئاً بالدعم النفسي.\n"
            "3. بما أن الطالب يستمع إليك (صوتياً فقط)، اجعل جملك قصيرة، مرتبة، وتجنب الجداول أو الرموز المعقدة التي يصعب سماعها.\n"
            "4. بعد شرح أي فكرة، اطرح سؤالاً بسيطاً للتأكد من فهم الطالب قبل الانتقال للنقطة التالية.\n"
            "5. اعتمد تماماً على المعلومات المقدمة لك في السياق للإجابة."
        )

    def generate_response(self, user_query, retrieved_context=""):
        """
        Sends the system prompt, retrieved RAG context, and user query to Ollama.
        """
        # Constructing the prompt by combining the RAG knowledge and the student's question
        full_user_content = f"السياق التعليمي المتوفر:\n{retrieved_context}\n\nسؤال الطالب: {user_query}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_user_content}
            ],
            "stream": False  # Set to False for now to make testing and debugging easier
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}\nMake sure Ollama is running (`ollama run qwen2.5:7b`)"

# --- Test Block ---
if __name__ == "__main__":
    # Ensure your Ollama application is running in the background before executing this!
    print("Initializing Pedagogical Agent...")
    tutor = PedagogicalAgent()
    
    # Fake test context (Simulating what your ChromaDB will eventually return)
    sample_context = "الشمس هي النجم المركزي للمجموعة الشمسية. وهي تتكون من غازات ساخنة وتمد الأرض بالضوء والحرارة."
    sample_query = "هي الشمس عبارة عن إيه وبيدير حواليها إيه؟"
    
    print("\nSending test prompt to Qwen2.5...")
    ai_response = tutor.generate_response(user_query=sample_query, retrieved_context=sample_context)
    
    print("\n--- Agent Response ---")
    print(ai_response)