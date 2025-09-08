# Information Retreival 學習日誌
## 階段一：文字預處理
1. 資料載入、轉換為小寫 (Lowercasing)、移除多餘空白 (Whitespace Removal)
1. 斷詞 (Tokenization): 使用 nltk.word_tokenize 將句子分解成一個個的詞彙
1. 拼字校正 (Spelling Correction): 使用 pyspellchecker 函式庫修正
1. 移除停用詞 (Stopword Removal)、移除標點符號 (Punctuation Removal): 使用正規表示式 RegexpTokenizer(r"\w+")
1. 詞形還原 (Lemmatization): 使用 WordNetLemmatizer還原為基本形
1. 詞幹提取 (Stemming): 使用 PorterStemmer 將單字簡化為其詞幹
1. 移除 HTML 標籤與網址 (Tag/URL Removal)、儲存結果

## 階段二：建立倒排索引
#### 一個字典結構，其中：鍵 (Key)是處理後的詞彙 (Token)。值 (Value)是一個列表，包含所有出現過該詞彙的文件 ID。
1. 建立一個空的字典 inverted_index = {}
1. 使用 os.listdir 讀取 test-collection/docs 資料夾中的每一個 .txt 檔案
1. 對每個文件的內容進行階段一預處理
1. 遍歷文件中的每一個有效詞彙，如果詞彙首次出現，就在 inverted_index 中新增一個鍵值設為包含當前文件 ID 的列表
1. 詞彙已存在，則將當前文件 ID 添加到對應的值
1. 使用 pickle 模組將建立好的 inverted_index 字典序列化並儲存為 inverted_index.pkl

## 階段三：詞彙加權
#### 計算詞的重要性分數，一個詞越稀有其 IDF 值越高
1. 使用 pickle.load 讀取上一階段儲存的 inverted_index.pkl
1. 根據 IDF 公式，使用 math.log10() 計算每個詞彙的 IDF 分數
1. 建立了一個 TF-IDF 索引存成一個巢狀字典 {term: {doc_id: tf_idf_score}}
1. 分別將 idf 字典和 tf_idf_index 字典儲存為 idf.pkl 和 tf_idf_index.pkl

## 階段四：搜尋與排序(Okapi BM25)
#### 傳統但有效的搜尋演算法: 透過 BM25 公式計算每份文件的相關性分數
1. 載入 inverted_index.pkl 和 idf.pkl
1. 對使用者輸入的查詢字串進行階段1的文字預處理
1. BM25公式結合了 IDF、TF 和文件長度懲罰，為每個文件計算一個總分
1. 將計算出的文件分數由高到低排序，並回傳排名最前面的文件

## 階段四：搜尋與排序(Sentence Transformer)
#### 深度學習的現代搜尋方法，理解語意而非僅用關鍵字
1. 使用 sentence-transformers 函式庫載入一個預訓練好的模型，例如 'all-MiniLM-L6-v2'
1. 準備文件集 (Corpus): 讀取所有 .txt 檔案的原始、未經處理的內容到一個列表中
1. 使用 model.encode(corpus) 將所有文件內容轉換成一個嵌入向量矩陣 corpus_embeddings。這一步取代了所有的預處理和索引建立過程
1. 輸入查詢時，使用同一個模型將查詢字串也轉換成一個向量 query_embedding
1. 使用 util.cos_sim() 計算查詢向量與所有文件向量的餘弦相似度分數
1. 相似度分數最高的前 k 個文件並回傳
