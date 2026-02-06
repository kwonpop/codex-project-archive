"""
Tkinter 기반 텍스트 분류기 생성 프로그램
--------------------------------------
사용자가 직접 문장 + 라벨 데이터를 입력하고,
즉석에서 머신러닝 모델(CountVectorizer + MultinomialNB)을 학습해
새로운 문장을 분류해볼 수 있는 교육용 예제입니다.

실행 방법:
    python text_classifier_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox

# scikit-learn 구성 요소
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class TextClassifierApp:
    """텍스트 분류기 GUI 애플리케이션 클래스"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("직접 만드는 텍스트 분류기")
        self.root.geometry("760x620")

        # ------------------------------
        # 내부 상태(데이터/모델)
        # ------------------------------
        # 사용자가 추가한 학습용 문장/라벨을 저장하는 리스트
        self.train_texts = []
        self.train_labels = []

        # 벡터화기 + 분류기 객체
        # 학습 전에는 실제 모델이 준비되지 않은 상태이므로 is_trained=False
        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        self.is_trained = False

        # 라벨 선택용 값 (긍정/부정/중립)
        self.selected_label = tk.StringVar(value="긍정")

        # 분류 결과 표시용 문자열 변수
        self.result_var = tk.StringVar(value="아직 분류 결과가 없습니다.")

        # UI를 구성하는 메서드 호출
        self.build_ui()

    def build_ui(self):
        """화면에 필요한 모든 위젯(입력창, 버튼, 리스트 등)을 배치합니다."""

        # ===== 1) 학습 데이터 입력 영역 =====
        input_frame = ttk.LabelFrame(self.root, text="1. 학습 데이터 입력", padding=12)
        input_frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(input_frame, text="문장 입력:").grid(row=0, column=0, sticky="w")

        # 학습 데이터용 문장 입력창
        self.train_sentence_entry = ttk.Entry(input_frame, width=75)
        self.train_sentence_entry.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)

        # 라벨 선택 (콤보박스)
        ttk.Label(input_frame, text="라벨 선택:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.label_combo = ttk.Combobox(
            input_frame,
            textvariable=self.selected_label,
            values=["긍정", "부정", "중립"],
            state="readonly",
            width=10,
        )
        self.label_combo.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        # 데이터 추가 버튼
        add_button = ttk.Button(input_frame, text="데이터 추가", command=self.add_training_data)
        add_button.grid(row=2, column=2, sticky="e", pady=(6, 0))

        input_frame.columnconfigure(0, weight=1)

        # ===== 2) 현재 학습 데이터 목록 표시 영역 =====
        list_frame = ttk.LabelFrame(self.root, text="2. 현재 학습 데이터 목록", padding=12)
        list_frame.pack(fill="both", expand=True, padx=12, pady=10)

        # 학습 데이터(문장 + 라벨)를 보여줄 리스트박스
        self.data_listbox = tk.Listbox(list_frame, height=12)
        self.data_listbox.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.data_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.data_listbox.config(yscrollcommand=scrollbar.set)

        # ===== 3) 모델 학습 버튼 영역 =====
        train_frame = ttk.Frame(self.root, padding=12)
        train_frame.pack(fill="x", padx=12, pady=2)

        train_button = ttk.Button(train_frame, text="모델 학습", command=self.train_model)
        train_button.pack(side="left")

        self.train_status_label = ttk.Label(train_frame, text="모델 상태: 아직 학습 전")
        self.train_status_label.pack(side="left", padx=12)

        # ===== 4) 테스트 문장 분류 영역 =====
        test_frame = ttk.LabelFrame(self.root, text="3. 테스트 문장 분류", padding=12)
        test_frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(test_frame, text="테스트할 문장 입력:").grid(row=0, column=0, sticky="w")

        # 분류용 테스트 문장 입력창
        self.test_sentence_entry = ttk.Entry(test_frame, width=75)
        self.test_sentence_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)

        # 분류 실행 버튼
        classify_button = ttk.Button(test_frame, text="분류하기", command=self.classify_text)
        classify_button.grid(row=2, column=0, sticky="w", pady=(4, 0))

        # 분류 결과 표시 라벨
        result_label = ttk.Label(
            test_frame,
            textvariable=self.result_var,
            foreground="blue",
            font=("맑은 고딕", 11, "bold"),
        )
        result_label.grid(row=2, column=1, sticky="w", padx=10, pady=(4, 0))

        test_frame.columnconfigure(0, weight=1)

        # Enter 키로 데이터 추가/분류를 편하게 사용할 수 있도록 바인딩
        self.train_sentence_entry.bind("<Return>", lambda event: self.add_training_data())
        self.test_sentence_entry.bind("<Return>", lambda event: self.classify_text())

    def add_training_data(self):
        """입력된 문장과 라벨을 내부 학습 데이터 목록에 추가합니다."""
        sentence = self.train_sentence_entry.get().strip()
        label = self.selected_label.get()

        # 빈 문장은 학습 데이터로 사용할 수 없으므로 경고
        if not sentence:
            messagebox.showwarning("입력 오류", "학습용 문장을 먼저 입력해주세요.")
            return

        # 내부 데이터 리스트에 추가
        self.train_texts.append(sentence)
        self.train_labels.append(label)

        # 화면 리스트박스에도 즉시 반영
        index = len(self.train_texts)
        self.data_listbox.insert(tk.END, f"{index}. [{label}] {sentence}")

        # 입력창 비우고 다시 포커스
        self.train_sentence_entry.delete(0, tk.END)
        self.train_sentence_entry.focus_set()

        # 새 데이터가 추가되었으므로 기존 모델은 최신 데이터 기준으로 재학습 필요
        self.is_trained = False
        self.train_status_label.config(text="모델 상태: 데이터 변경됨 (재학습 필요)")

    def train_model(self):
        """현재까지 입력된 데이터로 CountVectorizer + MultinomialNB 모델을 학습합니다."""

        # 최소 2개 이상의 라벨 데이터가 있어야 학습 의미가 큼
        if len(self.train_texts) < 2:
            messagebox.showwarning("데이터 부족", "최소 2개 이상의 학습 데이터를 추가해주세요.")
            return

        # 라벨 종류가 2개 이상이어야 분류 모델 학습이 가능
        unique_labels = set(self.train_labels)
        if len(unique_labels) < 2:
            messagebox.showwarning(
                "라벨 부족",
                "서로 다른 라벨이 최소 2종류 이상 필요합니다.\n"
                "예: 긍정/부정 또는 긍정/중립",
            )
            return

        # 1) 텍스트를 숫자 벡터로 변환
        # fit_transform: 현재 학습 데이터로 어휘 사전을 만들고 벡터화까지 수행
        x_train = self.vectorizer.fit_transform(self.train_texts)

        # 2) 나이브 베이즈 분류기 학습
        self.classifier.fit(x_train, self.train_labels)

        self.is_trained = True
        self.train_status_label.config(
            text=f"모델 상태: 학습 완료 (데이터 {len(self.train_texts)}개, 라벨 {len(unique_labels)}종류)"
        )
        messagebox.showinfo("학습 완료", "모델 학습이 완료되었습니다!")

    def classify_text(self):
        """테스트 문장을 현재 학습된 모델로 분류하고 결과를 화면에 표시합니다."""

        # 모델이 아직 학습되지 않았다면 먼저 학습을 유도
        if not self.is_trained:
            messagebox.showwarning("학습 필요", "먼저 '모델 학습' 버튼을 눌러 학습을 진행해주세요.")
            return

        test_sentence = self.test_sentence_entry.get().strip()
        if not test_sentence:
            messagebox.showwarning("입력 오류", "분류할 테스트 문장을 입력해주세요.")
            return

        # 학습 때 사용한 동일 vectorizer로 테스트 문장을 벡터화
        x_test = self.vectorizer.transform([test_sentence])

        # 예측 결과 (라벨 문자열) 추출
        predicted_label = self.classifier.predict(x_test)[0]

        # (선택) 각 라벨 확률도 함께 보고 싶다면 predict_proba 사용 가능
        # MultinomialNB는 predict_proba를 지원함
        probabilities = self.classifier.predict_proba(x_test)[0]
        class_names = self.classifier.classes_
        prob_text_parts = [f"{cls}: {prob:.2f}" for cls, prob in zip(class_names, probabilities)]
        prob_text = ", ".join(prob_text_parts)

        self.result_var.set(f"예측 라벨: {predicted_label}  |  확률: {prob_text}")


def main():
    """프로그램 시작점"""
    root = tk.Tk()
    app = TextClassifierApp(root)
    root.mainloop()


# 스크립트를 직접 실행했을 때만 main()이 호출되도록 처리
if __name__ == "__main__":
    main()
