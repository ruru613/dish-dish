import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from fastai.vision.all import *
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import base64
import random
import matplotlib.pyplot as plt
import time
import shutil
import torch
import pickle
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(page_title="食堂菜品识别系统", layout="wide")

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化会话状态
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = int(time.time() * 1000) % 1000000
if 'current_page' not in st.session_state:
    st.session_state.current_page = "首页"
if 'collab_model' not in st.session_state:
    st.session_state.collab_model = None

# 文件路径配置
RATINGS_FILE = Path(__file__).parent / '评分数据.xlsx'
BACKUP_DIR = Path(__file__).parent / 'ratings_backups'
DISHES_FILE = Path(__file__).parent / "菜品介绍.xlsx"
MODEL_PATH = Path(__file__).parent / "dish.pkl"  # 模型路径

# 创建备份目录
BACKUP_DIR.mkdir(exist_ok=True)

# 安全加载模型（修复路径和pickle问题）
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    
    logger.info(f"开始加载模型: {MODEL_PATH}")
    
    try:
        # 方案1：使用 Learner.load 替代 load_learner（更安全）
        # 注意：需要重新创建与训练时相同的DataLoaders
        data = ImageDataLoaders.from_folder(
            path=Path(__file__).parent,  # 假设图片在当前目录
            valid_pct=0.2,
            item_tfms=Resize(224),
            batch_tfms=aug_transforms(),
            bs=32
        )
        learn = vision_learner(data, resnet34, metrics=error_rate)
        learn.load(str(MODEL_PATH.with_suffix('')))  # 加载权重
        logger.info("模型加载成功 (方法1)")
        return learn
        
    except Exception as e1:
        logger.warning(f"方法1加载失败: {e1}. 尝试方法2...")
        try:
            # 方案2：使用 torch.load（更底层）
            with open(MODEL_PATH, 'rb') as f:
                model = torch.load(f, map_location='cpu', pickle_module=pickle)
            logger.info("模型加载成功 (方法2)")
            return model
        except Exception as e2:
            logger.error(f"方法2加载失败: {e2}. 尝试方法3...")
            try:
                # 方案3：使用 load_learner 但转为字符串路径
                model = load_learner(str(MODEL_PATH))
                logger.info("模型加载成功 (方法3)")
                return model
            except Exception as e3:
                error_msg = f"模型加载失败:\n方法1错误: {e1}\n方法2错误: {e2}\n方法3错误: {e3}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

# 加载模型（仅一次）
try:
    model = load_model()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# 加载菜品信息
try:
    dishes_df = pd.read_excel(DISHES_FILE)
    
    # 构建菜品ID映射
    dish_names = model.dls.vocab
    dish_id_map = {}
    for idx, row in dishes_df.iterrows():
        dish_name = row['dish_name']
        if dish_name in dish_names:
            dish_id_map[dish_name] = row.get('dish_id', idx + 1)
    
    # 验证映射完整性
    missing_dishes = [d for d in dish_names if d not in dish_id_map]
    if missing_dishes:
        st.warning(f"警告: 菜品信息表中缺少以下模型类别: {', '.join(missing_dishes)}")
        logger.warning(f"菜品信息表中缺少以下模型类别: {', '.join(missing_dishes)}")
        
except Exception as e:
    st.error(f"菜品信息加载失败: {e}")
    logger.error(f"菜品信息加载失败: {e}")
    st.stop()

# 辅助函数
def predict_dish(image):
    """使用模型预测菜品"""
    try:
        img = PILImage.create(image)
        pred, pred_idx, probs = model.predict(img)
        
        if pred not in dish_names:
            st.warning(f"异常预测结果: {pred} 不在模型类别列表中")
            logger.warning(f"异常预测结果: {pred} 不在模型类别列表中")
            pred = dish_names[np.argmax(probs)]
            st.info(f"已自动更正为最可能类别: {pred}")
            logger.info(f"已自动更正为最可能类别: {pred}")
            
        return pred, probs[pred_idx].item(), probs
    except Exception as e:
        st.error(f"预测失败: {e}")
        logger.error(f"预测失败: {e}")
        return None, 0, None

def display_dish_info(dish_name):
    """获取菜品详细信息"""
    if dish_name not in dish_id_map:
        return {
            "名称": dish_name,
            "菜系": "未知",
            "口味": "未知",
            "卡路里": "未知",
            "描述": "暂无详细信息",
            "推荐人群": "未知",
            "禁忌人群": "未知",
            "image": None
        }
        
    dish_info = dishes_df[dishes_df['dish_name'] == dish_name].iloc[0]
    return {
        "名称": dish_name,
        "菜系": dish_info['cuisine'],
        "口味": dish_info['taste'],
        "卡路里": f"{dish_info['calorie']}大卡每100克",
        "描述": dish_info['description'],
        "推荐人群": dish_info['recommended population'],
        "禁忌人群": dish_info['contraindicated population'],
        "image": dish_info.get('image', None)
    }

def set_page_style():
    """设置页面样式"""
    st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #FF6B6B;
        margin: 20px 0;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .rating-stars {
        color: #FFD700;
        font-size: 24px;
    }
    .recommendation-card {
        border-left: 4px solid #FF6B6B;
        padding-left: 15px;
        margin-bottom: 15px;
    }
    .highlight {
        color: #FF6B6B;
        font-weight: bold;
    }
    .error-message {
        color: red;
        font-weight: bold;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        color: #FF6B6B;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def get_download_link(df, filename):
    """生成下载链接"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下载评分数据</a>'
        return href
    except Exception as e:
        st.warning(f"生成下载链接失败: {e}")
        return None

def backup_ratings():
    """备份评分数据"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{BACKUP_DIR}/ratings_{timestamp}.xlsx"
        if RATINGS_FILE.exists():
            shutil.copy2(RATINGS_FILE, backup_path)
            logger.info(f"评分数据备份成功: {backup_path}")
            return backup_path
        return None
    except Exception as e:
        st.warning(f"评分数据备份失败: {e}")
        logger.error(f"评分数据备份失败: {e}")
        return None

def load_all_ratings():
    """安全加载评分数据"""
    try:
        if RATINGS_FILE.exists():
            return pd.read_excel(RATINGS_FILE)
        return pd.DataFrame(columns=['user_id', 'dish_id', 'rating', 'timestamp'])
    except Exception as e:
        st.warning(f"评分数据文件损坏，已创建空数据: {e}")
        logger.error(f"评分数据文件损坏: {e}")
        return pd.DataFrame(columns=['user_id', 'dish_id', 'rating', 'timestamp'])

def save_rating_safely(user_id, dish_id, rating):
    """安全保存评分数据"""
    # 新增异常值校验
    if not (1 <= rating <= 5):
        return False, "评分需在1-5星范围内,无法保存"
    if dish_id not in dish_id_map.values():
        return False, "无效的菜品ID,无法保存评分"
    
    new_rating = pd.DataFrame({
        'user_id': [user_id],
        'dish_id': [dish_id],
        'rating': [rating],
        'timestamp': [pd.Timestamp.now()]
    })
    
    try:
        backup_path = backup_ratings()
        if backup_path:
            st.info(f"已创建评分数据备份: {backup_path}")
            
        existing_data = load_all_ratings()
        combined_data = pd.concat([existing_data, new_rating], ignore_index=True)
        combined_data = combined_data.sort_values('timestamp', ascending=False)
        combined_data = combined_data.drop_duplicates(subset=['user_id', 'dish_id'], keep='first')
        
        combined_data.to_excel(RATINGS_FILE, index=False)
        user_ratings = combined_data[combined_data['user_id'] == user_id].copy()
        st.session_state.user_ratings = user_ratings.to_dict('records')
        
        logger.info(f"用户 {user_id} 对菜品 {dish_id} 评分 {rating} 保存成功")
        return True, "评分保存成功"
    
    except Exception as e:
        logger.error(f"评分保存失败: {e}")
        return False, f"评分保存失败: {str(e)}"

def load_collaborative_filtering_model():
    """加载协同过滤模型"""
    try:
        if Path(RATINGS_FILE).exists():
            data_df = load_all_ratings()
            
            if len(data_df) < 10:
                st.warning("评分数据不足，将使用基础推荐")
                logger.info("评分数据不足，使用基础推荐")
                return None
                
            reader = Reader(line_format='user item rating', rating_scale=(1, 5))
            data = Dataset.load_from_df(data_df[['user_id', 'dish_id', 'rating']], reader)
            trainset = data.build_full_trainset()
            
            algo = SVD(random_state=42, n_factors=100, n_epochs=5)
            algo.fit(trainset)
            logger.info("协同过滤模型加载成功")
            return algo
        else:
            st.warning("评分数据文件不存在，将使用基础推荐")
            logger.info("评分数据文件不存在，使用基础推荐")
            return None
    except Exception as e:
        st.warning(f"协同过滤模型加载失败，将使用基础推荐: {e}")
        logger.error(f"协同过滤模型加载失败: {e}")
        return None

# 页面函数
def home_page():
    """首页"""
    st.markdown('<div class="centered-title">🍱 食堂菜品识别系统</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    这是一个基于协同过滤算法的食堂菜品识别与推荐系统。您可以上传菜品图片，系统将识别菜品并为您提供菜品详细信息，在您食用过后可以对菜品进行评分，
    评分后系统会根据您的口味偏好为您推荐其他菜品,祝您用餐愉快🍽️🍽️🍽️!       当前用户ID: <span class='highlight'>{st.session_state.user_id}</span>
    """, unsafe_allow_html=True)
    
    st.info("请通过左侧导航栏选择功能模块")

def dish_recognition_page():
    """菜品识别页面"""
    st.markdown('<div class="centered-title">🍽️ 菜品识别</div>', unsafe_allow_html=True)
    
    st.subheader("上传菜品图片")
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="上传的菜品图片", use_container_width=True)
        
        with st.spinner("正在识别菜品..."):
            try:
                img = PILImage.create(uploaded_file)
                if img.size[0] < 50 or img.size[1] < 50:
                    st.warning("图片尺寸过小，可能影响识别准确率")
                    logger.warning("图片尺寸过小")
                    
                pred_dish, confidence, probs = predict_dish(img)
                if pred_dish:
                    st.markdown(f"识别结果: <span class='highlight'>{pred_dish}</span> (置信度: {confidence*100:.2f}%)", unsafe_allow_html=True)
                    
                    st.subheader("菜品介绍")
                    dish_info = display_dish_info(pred_dish)
                    for key, value in dish_info.items():
                        if key != "image":
                            st.markdown(f"**{key}:** {value}")
                    
                    st.subheader("识别概率分布")
                    valid_dishes = [dish for dish in dish_names if dish in dishes_df['dish_name'].values]
                    filtered_probs = [probs[i] for i, dish in enumerate(dish_names) if dish in valid_dishes]
                    
                    top5 = sorted(zip(valid_dishes, filtered_probs), key=lambda x: x[1], reverse=True)[:5]
                    labels = [item[0] for item in top5]
                    values = [item[1] for item in top5]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(labels, values, color='tomato')
                    ax.set_ylabel('概率')
                    ax.set_title('菜品识别概率分布')
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                    
                    # 评分功能
                    st.subheader("评价该菜品")
                    rating = st.slider("请给出评分 (1-5星)", 1, 5, 3)
                    
                    if st.button("提交评分"):
                        dish_id = dish_id_map.get(pred_dish, 0)
                        if dish_id == 0:
                            st.error(f"未找到菜品 {pred_dish} 的ID映射,评分失败")
                            logger.error(f"未找到菜品 {pred_dish} 的ID映射")
                            return
                            
                        success, message = save_rating_safely(
                            user_id=st.session_state.user_id,
                            dish_id=dish_id,
                            rating=rating
                        )
                        
                        if success:
                            st.success(f"感谢评分！您给{pred_dish}打了{rating}星")
                            st.markdown(f"<div class='rating-stars'>{'⭐' * rating}</div>", unsafe_allow_html=True)
                            logger.info(f"用户 {st.session_state.user_id} 给 {pred_dish} 评分 {rating}")
                            
                            # 重新加载协同过滤模型
                            st.session_state['collab_model'] = load_collaborative_filtering_model()
                            
                            # 提供下载链接
                            if st.session_state.user_ratings:
                                ratings_df = pd.DataFrame(st.session_state.user_ratings)
                                download_link = get_download_link(ratings_df, f'user_{st.session_state.user_id}_ratings.csv')
                                if download_link:
                                    st.markdown(download_link, unsafe_allow_html=True)
                        else:
                            st.error(message)
                            logger.error(f"评分提交失败: {message}")

            except Exception as e:
                st.error(f"图片处理出错: {e}")
                logger.error(f"图片处理出错: {e}")

def recommendation_page():
    """推荐页面"""
    st.markdown('<div class="centered-title">📋 菜品推荐</div>', unsafe_allow_html=True)
    
    if not st.session_state.user_ratings or len(st.session_state.user_ratings) == 0:
        st.warning("您还没有评分记录，请先识别并评价菜品，以便获取个性化推荐")
        return
    
    st.subheader("为您推荐菜品")
    with st.spinner("正在生成推荐..."):
        try:
            current_algo = st.session_state.get('collab_model', load_collaborative_filtering_model())
            
            if not current_algo:
                st.info("评分数据不足，使用基础推荐")
                logger.info("使用基础推荐")
                rated_dish_ids = [r['dish_id'] for r in st.session_state.user_ratings]
                recommended_dishes = dishes_df[~dishes_df['dish_id'].isin(rated_dish_ids)].sample(3)
                
                st.success("为您推荐（基础推荐）：")
                for i, row in recommended_dishes.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>**{i+1}. {row['dish_name']}** ({row['cuisine']})</h4>
                            <p>口味：{row['taste']} | 卡路里：{row['calorie']}大卡</p>
                            <p>描述：{row['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                new_user_ratings = pd.DataFrame(st.session_state.user_ratings)
                all_dish_ids = dishes_df['dish_id'].tolist()
                rated_dish_ids = new_user_ratings['dish_id'].tolist()
                unrated_dish_ids = [d for d in all_dish_ids if d not in rated_dish_ids]
                
                predictions = []
                for dish_id in unrated_dish_ids:
                    if dish_id in dishes_df['dish_id'].values:
                        pred = current_algo.predict(uid=st.session_state.user_id, iid=dish_id)
                        predictions.append((dish_id, pred.est))
                
                if predictions:
                    predictions_df = pd.DataFrame(predictions, columns=['dish_id', 'predicted_rating'])
                    recommendations = pd.merge(
                        predictions_df,
                        dishes_df[['dish_id', 'dish_name', 'cuisine', 'taste', 'calorie', 'description']],
                        on='dish_id'
                    ).sort_values('predicted_rating', ascending=False)
                    
                    st.success("为您推荐（协同过滤）：")
                    for i, row in recommendations.head(3).iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>**{i+1}. {row['dish_name']}** ({row['cuisine']})</h4>
                                <p>预测评分：{row['predicted_rating']:.2f}星 | 口味：{row['taste']} | 卡路里：{row['calorie']}大卡</p>
                                <p>描述：{row['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("没有可推荐的菜品，请尝试评价更多菜品")
                    logger.info("没有可推荐的菜品")
                    
        except Exception as e:
            st.error(f"推荐生成失败: {e}")
            logger.error(f"推荐生成失败: {e}")

def rating_statistics_page():
    """评分统计页面"""
    st.markdown('<div class="centered-title">📊 评分统计</div>', unsafe_allow_html=True)
    
    if not st.session_state.user_ratings or len(st.session_state.user_ratings) == 0:
        st.warning("您还没有评分记录")
        return
    
    ratings_df = pd.DataFrame(st.session_state.user_ratings)
    
    st.subheader("评分分布")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)
    
    st.subheader("您最喜欢的菜品")
    if 'dish_id' in ratings_df.columns and 'dish_name' in dishes_df.columns:
        most_liked = ratings_df.groupby('dish_id')['rating'].mean().nlargest(3)
        for dish_id, score in most_liked.items():
            try:
                dish_name = dishes_df[dishes_df['dish_id'] == dish_id]['dish_name'].iloc[0]
                st.markdown(f"- {dish_name}: {score:.2f}星")
            except:
                st.markdown(f"- 未知菜品 (ID: {dish_id}): {score:.2f}星")

def test_page():
    """测试页面"""
    st.markdown('<div class="centered-title">🧪 系统测试</div>', unsafe_allow_html=True)
    
    st.subheader("模型测试")
    
    test_image = st.file_uploader("上传测试图片", type=["jpg", "png", "jpeg"])
    
    if test_image:
        with st.spinner("正在测试模型..."):
            try:
                img = PILImage.create(test_image)
                pred, confidence, probs = predict_dish(img)
                
                if pred:
                    st.markdown(f"**预测结果**: {pred} (置信度: {confidence*100:.2f}%)")
                    
                    # 显示前5个预测结果
                    st.subheader("Top 5 预测")
                    top5 = sorted(zip(dish_names, probs), key=lambda x: x[1], reverse=True)[:5]
                    for i, (dish, prob) in enumerate(top5):
                        st.markdown(f"{i+1}. {dish}: {prob*100:.2f}%")
                    
                    # 显示图片
                    st.image(img, caption="测试图片", use_container_width=True)
                    
                else:
                    st.error("模型预测失败")
                    
            except Exception as e:
                st.error(f"测试失败: {e}")
                logger.error(f"测试失败: {e}")

# 主程序
def main():
    try:
        set_page_style()
        
        # 侧边栏导航
        st.sidebar.markdown('<div class="sidebar-title">导航菜单</div>', unsafe_allow_html=True)
        page_options = ["首页", "菜品识别", "菜品推荐", "评分统计", "系统测试"]
        selected_page = st.sidebar.radio("选择页面", page_options)
        
        # 更新当前页面状态
        st.session_state.current_page = selected_page
        
        # 显示对应页面
        if selected_page == "首页":
            home_page()
        elif selected_page == "菜品识别":
            dish_recognition_page()
        elif selected_page == "菜品推荐":
            recommendation_page()
        elif selected_page == "评分统计":
            rating_statistics_page()
        elif selected_page == "系统测试":
            test_page()
        
        # 页脚
        st.markdown("---")
        st.write("食堂菜品识别系统 🍽️ | 版本 1.0.0")
        
    except Exception as e:
        st.error(f"系统运行出错: {e}")
        logger.critical(f"系统运行出错: {e}", exc_info=True)

if __name__ == "__main__":
    main()