
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
app_dir = Path(__file__).parent.parent
root_dir = app_dir.parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(root_dir))

# Inject styling and load shared data
from styles import inject_page_css
from ml_utils import load_ml_data, clean_review_text, classify_price_sentiment_rule_based, aggregate_product_sentiment

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
inject_page_css()

st.title("💬 Sentiment Analysis")

try:
    df_feat, feature_cols, reviews_raw = load_ml_data()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["Sentiment (Rule-Based)", "Sentiment (ML Model)"])

# --------------------------------------------------------------------------------
# TAB 1: RULE-BASED SENTIMENT (COPIED EXACTLY)
# --------------------------------------------------------------------------------
with tab1:
    st.header("💬 Price Sentiment Analysis (Rule-Based)")
    st.markdown("""
    *Phân tích cảm xúc khách hàng về giá từ reviews bằng keyword matching.*
    
    **Cải tiến:**
    - ✅ Xử lý negation: "không đắt" ≠ negative
    - ✅ Ratio-based ranking thay vì count
    - ✅ Threshold tối thiểu mentions
    - ✅ Price mention rate metric
    """)
    
    st.divider()
    
    # Load reviews with text
    @st.cache_data(ttl=3600)
    def prepare_reviews_sentiment(reviews_df, products_df):
        reviews = reviews_df.copy()
        products = products_df[['product_id', 'product_name', 'brand_name', 'avg_final_price']].copy()
        reviews = reviews.merge(products, on='product_id', how='left')
        reviews = reviews[reviews['review_content'].notna() & (reviews['review_content'].str.len() > 10)]
        return reviews
    
    reviews_df = prepare_reviews_sentiment(reviews_raw, df_feat)
    st.success(f"✅ Đã tải {len(reviews_df):,} reviews có nội dung")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        min_mentions = st.slider("Minimum price mentions", 3, 20, 5, key="t3_min_mentions")
    with col2:
        run_sent = st.button("🚀 Phân Tích Price Sentiment", type="primary", key="t3_run")
    
    # Show keywords
    with st.expander("🔑 Từ khóa sử dụng"):
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Positive:**")
            st.write("xứng đáng, xứng tiền, rẻ, hợp lý, giá tốt, giá rẻ, đáng tiền, phải chăng, tiết kiệm, hời, giá mềm, giá ok")
        with col2:
            st.error("**Negative:**")
            st.write("đắt, mắc, chát, đắt quá, giá cao, quá đắt, đắt đỏ, không xứng tiền, lãng phí")
        st.info("**Negation handling:** 'không đắt', 'không mắc' → không tính là negative")
    
    # PERSISTENCE: Run if button clicked OR results exist
    if run_sent or ('rule_result_df' in st.session_state and 'rule_prod_df' in st.session_state):
        if run_sent:
            with st.spinner("Đang phân tích sentiment..."):
                # Clean text
                reviews_df['clean_text'] = reviews_df['review_content'].apply(clean_review_text)
                
                # Classify
                reviews_df['price_sentiment'] = reviews_df['clean_text'].apply(
                    lambda x: classify_price_sentiment_rule_based(x, handle_negation=True)
                )
                
                # Aggregate per product
                product_sentiment = aggregate_product_sentiment(
                    reviews_df, 
                    sentiment_col='price_sentiment',
                    min_mentions=min_mentions
                )
                
                # Merge with price info
                product_sentiment = product_sentiment.merge(
                    df_feat[['product_id', 'avg_final_price', 'avg_discount_percent', 'stockout_rate']],
                    on='product_id', how='left'
                )
                
                # SAVE TO SESSION
                st.session_state['rule_result_df'] = reviews_df
                st.session_state['rule_prod_df'] = product_sentiment
                
        # DISPLAY LOGIC
        if 'rule_result_df' in st.session_state:
            reviews_df = st.session_state['rule_result_df']
            product_sentiment = st.session_state['rule_prod_df']
            
            st.success("✅ Phân tích hoàn tất!")
            
            st.divider()
            
            st.subheader("📊 Tổng Quan")
            
            sentiment_counts = reviews_df['price_sentiment'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("😊 Positive", sentiment_counts.get('Positive', 0))
            col2.metric("😞 Negative", sentiment_counts.get('Negative', 0))
            col3.metric("😐 Neutral", sentiment_counts.get('Neutral', 0))
            col4.metric("📝 No Mention", sentiment_counts.get('No_Mention', 0))
            
            # Pie chart
            price_mentions = reviews_df[reviews_df['price_sentiment'] != 'No_Mention']
            if len(price_mentions) > 0:
                fig_pie = px.pie(
                    values=price_mentions['price_sentiment'].value_counts().values,
                    names=price_mentions['price_sentiment'].value_counts().index,
                    title="Phân bố Sentiment về Giá (chỉ reviews có đề cập)",
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            st.subheader("📝 Sản Phẩm Cần Hành Động (Ratio-Based)")
            
            # Filter products with enough mentions
            products_enough = product_sentiment[
                product_sentiment['price_mentions_count'] >= min_mentions
            ].copy()
            
            st.info(f"📊 {len(products_enough)} sản phẩm có ≥{min_mentions} price mentions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚠️ CẦN GIẢM GIÁ (High Negative Ratio)**")
                neg_products = products_enough.nlargest(15, 'negative_ratio')
                if len(neg_products) > 0:
                    show_cols = ['product_name', 'brand_name', 'avg_final_price', 
                                'negative_ratio', 'positive_ratio', 'price_mention_rate']
                    neg_display = neg_products[show_cols].copy()
                    neg_display['negative_ratio'] = neg_display['negative_ratio'].apply(lambda x: f"{x:.1%}")
                    neg_display['positive_ratio'] = neg_display['positive_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    neg_display['price_mention_rate'] = neg_display['price_mention_rate'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(neg_display, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**💎 CÓ ROOM TĂNG GIÁ (High Positive Ratio)**")
                pos_products = products_enough.nlargest(15, 'positive_ratio')
                if len(pos_products) > 0:
                    show_cols = ['product_name', 'brand_name', 'avg_final_price', 
                                'positive_ratio', 'negative_ratio', 'price_mention_rate']
                    pos_display = pos_products[show_cols].copy()
                    pos_display['positive_ratio'] = pos_display['positive_ratio'].apply(lambda x: f"{x:.1%}")
                    pos_display['negative_ratio'] = pos_display['negative_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    pos_display['price_mention_rate'] = pos_display['price_mention_rate'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(pos_display, use_container_width=True, hide_index=True)
            
            st.divider()
            
            st.subheader("📄 Sample Reviews")
            
            tab_pos, tab_neg = st.tabs(["😊 Positive", "😞 Negative"])
            
            with tab_pos:
                pos_reviews = price_mentions[price_mentions['price_sentiment'] == 'Positive'].head(5)
                for _, row in pos_reviews.iterrows():
                    with st.expander(f"⭐ {row.get('rating_star', 'N/A')}/5 - {str(row.get('product_name', ''))[:40]}..."):
                        st.write(f"**Giá:** {row.get('avg_final_price', 0):,.0f}đ")
                        st.write(f"**Review:** {row['review_content']}")
            
            with tab_neg:
                neg_reviews = price_mentions[price_mentions['price_sentiment'] == 'Negative'].head(5)
                for _, row in neg_reviews.iterrows():
                    with st.expander(f"⭐ {row.get('rating_star', 'N/A')}/5 - {str(row.get('product_name', ''))[:40]}..."):
                        st.write(f"**Giá:** {row.get('avg_final_price', 0):,.0f}đ")
                        st.write(f"**Review:** {row['review_content']}")

# --------------------------------------------------------------------------------
# TAB 2: ML SENTIMENT (COPIED EXACTLY)
# --------------------------------------------------------------------------------
with tab2:
    st.header("🤖 ML Sentiment Analysis (Pre-trained)")
    st.markdown("""
    *Phân loại sentiment bằng model đã train sẵn cho tiếng Việt - KHÔNG cần train!*
    
    **Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment` (hỗ trợ 8 ngôn ngữ kể cả Việt)
    """)
    
    st.divider()
    
    st.subheader("📦 Dependencies Status")
    
    torch_available = False
    transformers_available = False
    
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
    except ImportError:
        torch_version = "Not installed"
    
    try:
        import transformers
        transformers_available = True
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "Not installed"
    
    col1, col2 = st.columns(2)
    with col1:
        if torch_available:
            st.success(f"✅ PyTorch: {torch_version}")
        else:
            st.error("❌ PyTorch: Not installed")
    with col2:
        if transformers_available:
            st.success(f"✅ Transformers: {transformers_version}")
        else:
            st.error("❌ Transformers: Not installed")
    
    if not (torch_available and transformers_available):
        st.error("""
        ⚠️ **Thiếu thư viện!**
        
        Cài đặt:
        ```bash
        pip install torch transformers sentencepiece
        ```
        """)
        st.stop()
    
    st.divider()
    
    st.subheader("📥 Load Pre-trained Model")
    
    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    
    @st.cache_resource
    def load_pretrained_sentiment():
        """Load pre-trained multilingual sentiment model"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        return tokenizer, model
    
    try:
        with st.spinner("Đang load model (lần đầu sẽ download ~1GB)..."):
            tokenizer, model = load_pretrained_sentiment()
        st.success(f"✅ Model loaded: `{MODEL_NAME}`")
        st.info("""
        **Về model này:**
        - XLM-RoBERTa fine-tuned trên 198M tweets
        - Hỗ trợ 8 ngôn ngữ: AR, EN, FR, DE, HI, IT, SP, PT (và hoạt động tốt với Việt)
        - 3 classes: Negative (0), Neutral (1), Positive (2)
        - **KHÔNG cần train thêm!**
        """)
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        model_loaded = False
        st.stop()
    
    st.divider()
    
    st.subheader("🔮 Predict Sentiment")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_reviews = st.slider("Min reviews per product", 3, 20, 5, key="pt_min_reviews")
    with col2:
        neg_threshold = st.slider("Neg ratio threshold", 0.3, 0.8, 0.5, key="pt_neg_thresh")
    with col3:
        pos_threshold = st.slider("Pos ratio threshold", 0.3, 0.8, 0.5, key="pt_pos_thresh")
    
    predict_button = st.button("🚀 Predict All Reviews", type="primary", key="pt_predict")
    
    # PERSISTENCE
    if predict_button or ('ml_predictions' in st.session_state and 'ml_sentiment_results' in st.session_state):
        if predict_button and model_loaded:
            import torch
            
            # Prepare reviews
            predict_reviews = reviews_raw.copy()
            predict_reviews = predict_reviews[
                predict_reviews['review_content'].notna() & 
                (predict_reviews['review_content'].str.len() > 10)
            ].copy()
            
            # Merge with product info
            predict_reviews = predict_reviews.merge(
                df_feat[['product_id', 'product_name', 'brand_name', 
                        'avg_final_price', 'avg_discount_percent', 'stockout_rate']],
                on='product_id', how='left'
            )
            
            st.write(f"📊 Predicting {len(predict_reviews)} reviews...")
            
            # Batch prediction
            ID2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            predictions = []
            confidences = []
            batch_size = 16
            
            progress = st.progress(0)
            
            with st.spinner("Đang predict..."):
                for i in range(0, len(predict_reviews), batch_size):
                    batch = predict_reviews.iloc[i:i+batch_size]
                    texts = [clean_review_text(str(t))[:512] for t in batch['review_content'].tolist()]
                    
                    try:
                        inputs = tokenizer(
                            texts, truncation=True, max_length=128,
                            padding=True, return_tensors='pt'
                        ).to(device)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            preds = outputs.logits.argmax(-1).cpu().numpy()
                            confs = probs.max(dim=-1).values.cpu().numpy()
                        
                        predictions.extend([ID2LABEL[p] for p in preds])
                        confidences.extend(confs.tolist())
                    except Exception as e:
                        # Fallback for failed batches
                        predictions.extend(['Neutral'] * len(batch))
                        confidences.extend([0.5] * len(batch))
                    
                    progress.progress(min(1.0, (i + batch_size) / len(predict_reviews)))
            
            predict_reviews['ml_sentiment'] = predictions
            predict_reviews['confidence'] = confidences
            
            # SAVE FOR TAB USE
            st.session_state['ml_predictions'] = predict_reviews
            
            # Product aggregation for dashboard
            product_agg = predict_reviews.groupby('product_id').agg({
                'product_name': 'first',
                'brand_name': 'first',
                'avg_final_price': 'first',
                'avg_discount_percent': 'first',
                'stockout_rate': 'first',
                'ml_sentiment': [
                    ('total', 'count'),
                    ('pos_count', lambda x: (x == 'Positive').sum()),
                    ('neg_count', lambda x: (x == 'Negative').sum()),
                    ('neu_count', lambda x: (x == 'Neutral').sum()),
                ]
            })
            product_agg.columns = ['product_name', 'brand_name', 'avg_final_price', 
                                  'avg_discount_percent', 'stockout_rate',
                                  'total', 'pos_count', 'neg_count', 'neu_count']
            product_agg = product_agg.reset_index()
            
            # Calculate ratios
            product_agg['pos_ratio'] = np.where(
                product_agg['total'] >= min_reviews,
                product_agg['pos_count'] / product_agg['total'],
                np.nan
            )
            product_agg['neg_ratio'] = np.where(
                product_agg['total'] >= min_reviews,
                product_agg['neg_count'] / product_agg['total'],
                np.nan
            )
            
            # Filter products with enough reviews
            product_enough = product_agg[product_agg['total'] >= min_reviews].copy()
            
            # SAVE to session_state
            st.session_state['ml_sentiment_results'] = product_enough.copy()
            
        # DISPLAY LOGIC
        if 'ml_predictions' in st.session_state and 'ml_sentiment_results' in st.session_state:
            predict_reviews = st.session_state['ml_predictions']
            product_enough = st.session_state['ml_sentiment_results']
            
            st.success(f"✅ Predicted {len(predict_reviews)} reviews!")
            
            st.divider()
            st.subheader("📊 Overall Sentiment Distribution")
            
            sent_counts = predict_reviews['ml_sentiment'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("😊 Positive", sent_counts.get('Positive', 0), 
                       f"{sent_counts.get('Positive', 0)/len(predict_reviews)*100:.1f}%")
            col2.metric("😐 Neutral", sent_counts.get('Neutral', 0),
                       f"{sent_counts.get('Neutral', 0)/len(predict_reviews)*100:.1f}%")
            col3.metric("😞 Negative", sent_counts.get('Negative', 0),
                       f"{sent_counts.get('Negative', 0)/len(predict_reviews)*100:.1f}%")
            
            # Pie chart
            fig_pie = px.pie(
                values=sent_counts.values,
                names=sent_counts.index,
                title="ML Sentiment Distribution",
                color=sent_counts.index,
                color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            st.subheader("📦 Product-Level Sentiment")
            
            st.info(f"📊 {len(product_enough)} products có ≥{min_reviews} reviews")
            st.success("💾 Kết quả đã được lưu để dùng trong Decision Dashboard!")
            
            st.divider()
            st.subheader("💰 Pricing Action Lists")
            
            stockout_threshold = 0.3
            neg_threshold_val = neg_threshold # Use slider value
            pos_threshold_val = pos_threshold # Use slider value
            
            # 1. SUPPLY CONSTRAINT FIRST
            supply_constrained = product_enough[
                product_enough['stockout_rate'] > stockout_threshold
            ].sort_values('neg_ratio', ascending=False).head(20)
            
            # 2. GIẢM GIÁ (high neg_ratio + good stock)
            reduce_price = product_enough[
                (product_enough['neg_ratio'] >= neg_threshold_val) &
                (product_enough['stockout_rate'] <= stockout_threshold)
            ].sort_values('neg_ratio', ascending=False).head(20)
            
            # 3. TĂNG GIÁ (high pos_ratio + low discount + good stock)
            median_discount = product_enough['avg_discount_percent'].median()
            increase_price = product_enough[
                (product_enough['pos_ratio'] >= pos_threshold_val) &
                (product_enough['avg_discount_percent'] < median_discount) &
                (product_enough['stockout_rate'] <= stockout_threshold)
            ].sort_values('pos_ratio', ascending=False).head(20)
            
            display_cols = ['product_name', 'brand_name', 'avg_final_price', 
                           'avg_discount_percent', 'stockout_rate', 'total', 
                           'pos_ratio', 'neg_ratio']
            
            # Display action lists
            st.markdown("**📦 1. FIX SUPPLY FIRST (Stockout cao)**")
            if len(supply_constrained) > 0:
                supply_display = supply_constrained[display_cols].copy()
                supply_display['pos_ratio'] = supply_display['pos_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                supply_display['neg_ratio'] = supply_display['neg_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(supply_display, use_container_width=True, hide_index=True)
                st.warning(f"⚠️ {len(supply_constrained)} sản phẩm cần restock trước!")
            else:
                st.success("✅ Không có sản phẩm stockout cao")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**⬇️ 2. GIẢM GIÁ (neg_ratio ≥ {neg_threshold_val:.0%})**")
                if len(reduce_price) > 0:
                    red_display = reduce_price[display_cols].copy()
                    red_display['pos_ratio'] = red_display['pos_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    red_display['neg_ratio'] = red_display['neg_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    st.dataframe(red_display, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có sản phẩm phù hợp")
            
            with col2:
                st.markdown(f"**⬆️ 3. CÓ ROOM TĂNG GIÁ (pos_ratio ≥ {pos_threshold_val:.0%})**")
                if len(increase_price) > 0:
                    inc_display = increase_price[display_cols].copy()
                    inc_display['pos_ratio'] = inc_display['pos_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    inc_display['neg_ratio'] = inc_display['neg_ratio'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                    st.dataframe(inc_display, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có sản phẩm phù hợp")
            
            st.divider()
            st.subheader("⚖️ So Sánh với Rule-Based")
            
            predict_reviews['rule_sentiment'] = predict_reviews['review_content'].apply(
                lambda x: classify_price_sentiment_rule_based(clean_review_text(str(x)), handle_negation=True)
            )
            
            # Only compare where rule-based found something
            comparable = predict_reviews[predict_reviews['rule_sentiment'] != 'No_Mention']
            
            if len(comparable) > 0:
                agreement = (comparable['rule_sentiment'] == comparable['ml_sentiment']).mean()
                st.metric("Agreement Rate", f"{agreement:.1%}")
                
                crosstab = pd.crosstab(comparable['rule_sentiment'], comparable['ml_sentiment'])
                st.dataframe(crosstab, use_container_width=True)
    
    st.divider()
    
    st.subheader("✍️ Thử Predict")
    
    user_text = st.text_area("Nhập review:", "Sản phẩm tốt, giá hợp lý, đáng mua", key="pt_user_input")
    
    if st.button("🔮 Predict", key="pt_single_predict"):
        if model_loaded:
            import torch
            
            ID2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            
            clean = clean_review_text(user_text)
            inputs = tokenizer(clean, return_tensors='pt', truncation=True, max_length=128)
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_id = outputs.logits.argmax(-1).item()
                confidence = probs[0][pred_id].item()
            
            pred = ID2LABEL[pred_id]
            
            if pred == 'Positive':
                st.success(f"😊 **Positive** (confidence: {confidence:.1%})")
            elif pred == 'Negative':
                st.error(f"😞 **Negative** (confidence: {confidence:.1%})")
            else:
                st.info(f"😐 **Neutral** (confidence: {confidence:.1%})")
