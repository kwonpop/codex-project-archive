"""
Tkinter 기반 텍스트 분류기 생성 프로그램
--------------------------------------
사용자가 직접 문장 + 라벨 데이터를 입력하고,
즉석에서 머신러닝 모델(CountVectorizer + MultinomialNB)을 학습해
새로운 문장을 분류해볼 수 있는 교육용 예제입니다.

추가 기능(개선):
- 예시 데이터 자동 추가
- 선택 데이터 삭제 / 전체 데이터 초기화
- 현재 학습 데이터 기준 간단 정확도 표시
- 분류 히스토리 표시

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
        self.root.geometry("840x760")

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

        # 학습 정확도(학습 데이터 기준) 표시용 문자열 변수
        self.accuracy_var = tk.StringVar(value="정확도: 아직 계산되지 않음")

        # UI를 구성하는 메서드 호출
        self.build_ui()

    def build_ui(self):
        """화면에 필요한 모든 위젯(입력창, 버튼, 리스트 등)을 배치합니다."""

        # ===== 1) 학습 데이터 입력 영역 =====
        input_frame = ttk.LabelFrame(self.root, text="1. 학습 데이터 입력", padding=12)
        input_frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(input_frame, text="문장 입력:").grid(row=0, column=0, sticky="w")

        # 학습 데이터용 문장 입력창
        self.train_sentence_entry = ttk.Entry(input_frame, width=90)
        self.train_sentence_entry.grid(row=1, column=0, columnspan=5, sticky="ew", pady=6)

        # 라벨 선택 (콤보박스)
        ttk.Label(input_frame, text="라벨 선택:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.label_combo = ttk.Combobox(
            input_frame,
            textvariable=self.selected_label,
            values=["긍정", "부정", "중립"],
            state="readonly",
            width=10,
        )
        self.label_combo.grid(row=2, column=1, sticky="w", padx=(8, 8), pady=(6, 0))

        # 데이터 추가 버튼
        add_button = ttk.Button(input_frame, text="데이터 추가", command=self.add_training_data)
        add_button.grid(row=2, column=2, sticky="w", pady=(6, 0))

        # 예시 데이터 넣기 버튼
        sample_button = ttk.Button(input_frame, text="예시 데이터 자동 추가", command=self.load_sample_data)
        sample_button.grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(6, 0))

        # 선택 삭제 버튼
        delete_button = ttk.Button(input_frame, text="선택 데이터 삭제", command=self.delete_selected_data)
        delete_button.grid(row=2, column=4, sticky="e", pady=(6, 0))

        input_frame.columnconfigure(0, weight=1)

        # ===== 2) 현재 학습 데이터 목록 표시 영역 =====
        list_frame = ttk.LabelFrame(self.root, text="2. 현재 학습 데이터 목록", padding=12)
        list_frame.pack(fill="both", expand=True, padx=12, pady=10)

        # 학습 데이터(문장 + 라벨)를 보여줄 리스트박스
        self.data_listbox = tk.Listbox(list_frame, height=11)
        self.data_listbox.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.data_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.data_listbox.config(yscrollcommand=scrollbar.set)

        # ===== 3) 모델 학습/초기화 버튼 영역 =====
        train_frame = ttk.Frame(self.root, padding=12)
        train_frame.pack(fill="x", padx=12, pady=2)

        train_button = ttk.Button(train_frame, text="모델 학습", command=self.train_model)
        train_button.pack(side="left")

        clear_button = ttk.Button(train_frame, text="전체 데이터 초기화", command=self.clear_all_data)
        clear_button.pack(side="left", padx=(8, 0))

        self.train_status_label = ttk.Label(train_frame, text="모델 상태: 아직 학습 전")
        self.train_status_label.pack(side="left", padx=12)

        accuracy_label = ttk.Label(train_frame, textvariable=self.accuracy_var, foreground="darkgreen")
        accuracy_label.pack(side="right")

        # ===== 4) 테스트 문장 분류 영역 =====
        test_frame = ttk.LabelFrame(self.root, text="3. 테스트 문장 분류", padding=12)
        test_frame.pack(fill="x", padx=12, pady=10)

        ttk.Label(test_frame, text="테스트할 문장 입력:").grid(row=0, column=0, sticky="w")

        # 분류용 테스트 문장 입력창
        self.test_sentence_entry = ttk.Entry(test_frame, width=90)
        self.test_sentence_entry.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)

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
        result_label.grid(row=2, column=1, columnspan=2, sticky="w", padx=10, pady=(4, 0))

        test_frame.columnconfigure(0, weight=1)

        # ===== 5) 예측 히스토리 표시 영역 =====
        history_frame = ttk.LabelFrame(self.root, text="4. 분류 히스토리", padding=12)
        history_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.history_listbox = tk.Listbox(history_frame, height=7)
        self.history_listbox.pack(fill="both", expand=True, side="left")

        history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_listbox.yview)
        history_scroll.pack(side="right", fill="y")
        self.history_listbox.config(yscrollcommand=history_scroll.set)

        # Enter 키로 데이터 추가/분류를 편하게 사용할 수 있도록 바인딩
        self.train_sentence_entry.bind("<Return>", lambda event: self.add_training_data())
        self.test_sentence_entry.bind("<Return>", lambda event: self.classify_text())

    def refresh_data_listbox(self):
        """내부 데이터(self.train_texts, self.train_labels)를 리스트박스에 다시 그립니다."""
        self.data_listbox.delete(0, tk.END)
        for idx, (text, label) in enumerate(zip(self.train_texts, self.train_labels), start=1):
            self.data_listbox.insert(tk.END, f"{idx}. [{label}] {text}")

    def set_dirty_model_status(self):
        """데이터가 바뀐 뒤 모델 재학습 필요 상태를 UI에 반영합니다."""
        self.is_trained = False
        self.train_status_label.config(text="모델 상태: 데이터 변경됨 (재학습 필요)")
        self.accuracy_var.set("정확도: 재학습 후 확인 가능")

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
        self.refresh_data_listbox()

        # 입력창 비우고 다시 포커스
        self.train_sentence_entry.delete(0, tk.END)
        self.train_sentence_entry.focus_set()

        # 새 데이터가 추가되었으므로 기존 모델은 최신 데이터 기준으로 재학습 필요
        self.set_dirty_model_status()

    def load_sample_data(self):
        """초보자 실습을 돕기 위해 기본 예시 데이터를 자동으로 넣습니다."""
        sample_rows = [
            # ===== 긍정 =====
            ("이 영화는 정말 감동적이고 좋았어요", "긍정"),
            ("서비스가 친절해서 기분이 좋았습니다", "긍정"),
            ("배송이 예상보다 빨라서 만족해요", "긍정"),
            ("음식이 신선하고 맛있어서 또 주문할래요", "긍정"),
            ("앱이 직관적이라 사용하기 편했습니다", "긍정"),
            ("설명이 자세해서 초보자도 따라 하기 쉬웠어요", "긍정"),
            ("가격 대비 성능이 정말 뛰어납니다", "긍정"),
            ("문의 답변이 빠르고 정확해서 좋았어요", "긍정"),
            ("포장이 깔끔하고 제품 상태가 완벽했어요", "긍정"),
            ("업데이트 후 속도가 더 빨라졌네요", "긍정"),
            ("강의 내용이 탄탄해서 큰 도움이 됐어요", "긍정"),
            ("직원들이 밝고 응대가 프로페셔널합니다", "긍정"),
            ("디자인이 예쁘고 마감도 고급스럽습니다", "긍정"),
            ("문제가 생겼는데 바로 해결해줘서 감동했어요", "긍정"),
            ("재구매 의사가 충분히 있습니다", "긍정"),
            ("배터리가 오래가서 외출할 때 편리해요", "긍정"),
            ("화질이 선명해서 영상 보는 재미가 있어요", "긍정"),
            ("세팅이 간단해서 설치가 금방 끝났어요", "긍정"),
            ("기능이 다양해서 활용도가 높습니다", "긍정"),
            ("전반적으로 기대 이상이라 매우 만족합니다", "긍정"),
            ("고객센터 안내가 친절하고 이해하기 쉬웠어요", "긍정"),
            ("반응 속도가 빨라서 스트레스가 없네요", "긍정"),
            ("마감 품질이 좋아서 오래 쓸 수 있겠어요", "긍정"),
            ("성능 테스트 결과가 매우 안정적이었습니다", "긍정"),
            ("문서가 잘 정리되어 있어 학습하기 편했습니다", "긍정"),

            # ===== 부정 =====
            ("음식 맛이 별로고 다시는 안 갈 것 같아요", "부정"),
            ("배송이 너무 늦어서 실망했습니다", "부정"),
            ("앱이 자주 멈춰서 사용하기 불편해요", "부정"),
            ("고객센터 연결이 안 돼서 답답했습니다", "부정"),
            ("가격이 비싼데 품질은 기대 이하여서 아쉬워요", "부정"),
            ("포장이 부실해서 제품이 손상되어 왔어요", "부정"),
            ("설명이 부족해서 설치에 시간이 너무 오래 걸렸어요", "부정"),
            ("업데이트 이후 오류가 더 많아졌습니다", "부정"),
            ("응대 태도가 불친절해서 기분이 나빴어요", "부정"),
            ("기능이 홍보와 달라서 실망했어요", "부정"),
            ("결제가 여러 번 실패해서 이용을 포기했습니다", "부정"),
            ("소음이 심해서 집중하기 어려워요", "부정"),
            ("화면 색감이 탁하고 선명하지 않습니다", "부정"),
            ("버그 때문에 중요한 작업을 날렸어요", "부정"),
            ("환불 절차가 복잡하고 처리도 느립니다", "부정"),
            ("재질이 약해서 금방 망가졌어요", "부정"),
            ("배터리가 너무 빨리 닳아서 불편합니다", "부정"),
            ("접속이 자주 끊겨서 신뢰하기 어렵네요", "부정"),
            ("강의가 중복 설명만 많고 핵심이 부족해요", "부정"),
            ("검색 결과가 부정확해서 원하는 정보를 못 찾았어요", "부정"),
            ("광고가 너무 많아서 사용 경험이 나쁩니다", "부정"),
            ("초기 설정이 복잡해서 진입 장벽이 높아요", "부정"),
            ("설치 파일이 손상되어 실행조차 안 됩니다", "부정"),
            ("약속한 일정이 계속 미뤄져서 신뢰가 떨어졌어요", "부정"),
            ("전반적으로 비용 대비 만족도가 낮습니다", "부정"),

            # ===== 중립 =====
            ("그냥 무난했고 특별한 느낌은 없었어요", "중립"),
            ("평범한 하루였고 큰 일은 없었습니다", "중립"),
            ("기본 기능은 동작하지만 장점도 단점도 뚜렷하지 않아요", "중립"),
            ("가격은 보통 수준이고 품질도 보통입니다", "중립"),
            ("설치 과정은 일반적인 수준이었습니다", "중립"),
            ("생각했던 것과 거의 비슷했습니다", "중립"),
            ("사용은 가능하지만 인상적이진 않았어요", "중립"),
            ("배송은 예정일에 맞춰 도착했습니다", "중립"),
            ("디자인은 취향에 따라 평가가 갈릴 것 같아요", "중립"),
            ("크게 불편하지도 아주 편하지도 않았습니다", "중립"),
            ("튜토리얼은 기본 내용 위주로 구성되어 있어요", "중립"),
            ("실사용 전후 차이는 크지 않았습니다", "중립"),
            ("필요한 기능은 있지만 추가 기능은 많지 않아요", "중립"),
            ("화질은 평균적인 수준으로 보입니다", "중립"),
            ("배터리 사용 시간은 스펙에 적힌 정도입니다", "중립"),
            ("지원 문서는 표준적인 형식으로 작성되어 있습니다", "중립"),
            ("이벤트 혜택은 평이한 편이었어요", "중립"),
            ("장기간 사용해봐야 정확한 판단이 가능할 것 같아요", "중립"),
            ("선호도에 따라 만족도가 달라질 수 있습니다", "중립"),
            ("특별히 추천하거나 비추천할 정도는 아닙니다", "중립"),
            ("성능은 일상적인 작업에는 무난합니다", "중립"),
            ("설명서 내용이 표준적이라 익숙했습니다", "중립"),
            ("처음 사용 시 큰 장벽은 없었습니다", "중립"),
            ("전반적으로 평균에 가까운 결과였습니다", "중립"),
            ("기능은 충분하지만 차별점은 크지 않습니다", "중립"),
        ]

        for text, label in sample_rows:
            self.train_texts.append(text)
            self.train_labels.append(label)

        self.refresh_data_listbox()
        self.set_dirty_model_status()
        messagebox.showinfo("예시 데이터 추가", f"예시 데이터 {len(sample_rows)}개를 추가했습니다.")

    def delete_selected_data(self):
        """리스트에서 선택한 학습 데이터를 삭제합니다."""
        selected = self.data_listbox.curselection()
        if not selected:
            messagebox.showwarning("선택 필요", "삭제할 데이터를 목록에서 선택해주세요.")
            return

        # 여러 개 선택된 경우를 고려해 뒤에서부터 삭제해야 인덱스가 안전함
        for index in reversed(selected):
            del self.train_texts[index]
            del self.train_labels[index]

        self.refresh_data_listbox()
        self.set_dirty_model_status()

    def clear_all_data(self):
        """학습 데이터와 예측 히스토리를 전체 초기화합니다."""
        if not self.train_texts and self.history_listbox.size() == 0:
            messagebox.showinfo("안내", "초기화할 데이터가 없습니다.")
            return

        confirmed = messagebox.askyesno("초기화 확인", "학습 데이터와 히스토리를 모두 지울까요?")
        if not confirmed:
            return

        self.train_texts.clear()
        self.train_labels.clear()

        self.data_listbox.delete(0, tk.END)
        self.history_listbox.delete(0, tk.END)

        self.result_var.set("아직 분류 결과가 없습니다.")
        self.train_status_label.config(text="모델 상태: 아직 학습 전")
        self.accuracy_var.set("정확도: 아직 계산되지 않음")
        self.is_trained = False

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

        # 학습 데이터에 대해 간단 정확도 계산(교육용 지표)
        train_accuracy = self.classifier.score(x_train, self.train_labels)

        self.train_status_label.config(
            text=f"모델 상태: 학습 완료 (데이터 {len(self.train_texts)}개, 라벨 {len(unique_labels)}종류)"
        )
        self.accuracy_var.set(f"정확도(학습 데이터 기준): {train_accuracy:.2%}")

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

        # 각 라벨 확률도 함께 표시
        probabilities = self.classifier.predict_proba(x_test)[0]
        class_names = self.classifier.classes_
        prob_text_parts = [f"{cls}: {prob:.2f}" for cls, prob in zip(class_names, probabilities)]
        prob_text = ", ".join(prob_text_parts)

        result_text = f"예측 라벨: {predicted_label}  |  확률: {prob_text}"
        self.result_var.set(result_text)

        # 히스토리 리스트에 최근 분류 기록을 추가
        self.history_listbox.insert(tk.END, f"입력: {test_sentence}")
        self.history_listbox.insert(tk.END, f"결과: {result_text}")
        self.history_listbox.insert(tk.END, "-" * 80)
        self.history_listbox.see(tk.END)


def main():
    """프로그램 시작점"""
    root = tk.Tk()
    app = TextClassifierApp(root)
    root.mainloop()


# 스크립트를 직접 실행했을 때만 main()이 호출되도록 처리
if __name__ == "__main__":
    main()
