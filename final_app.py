import streamlit as st
import pandas as pd
import plotly.express as px

def get_dataframe_from_excel():
    df = pd.read_excel('supermarket_sales.xlsx',
                       sheet_name='销售数据',
                       skiprows=1,
                       index_col='订单号'
                       )
    df['小时数'] = pd.to_datetime(df["时间"], format="%H:%M:%S").dt.hour
    return df

def add_sidebar_func(df):
    # 创建侧边栏
    with st.sidebar:
        st.header("请筛选数据：")
        
        # 城市筛选：空值时默认全选
        city_unique = df["城市"].unique()
        city = st.multiselect(
            "请选择城市：",
            options=city_unique,
            default=city_unique,  # 初始默认全选
        )
        if not city:
            city = city_unique
        
        # 顾客类型筛选：空值时默认全选
        customer_type_unique = df["顾客类型"].unique()
        customer_type = st.multiselect(
            "请选择顾客类型：",
            options=customer_type_unique,
            default=customer_type_unique,
        )
        if not customer_type:
            customer_type = customer_type_unique
        
        # 性别筛选：空值时默认全选
        gender_unique = df["性别"].unique()
        gender = st.multiselect(
            "请选择性别",
            options=gender_unique,
            default=gender_unique,
        )
        if not gender:
            gender = gender_unique
        
        # 修复核心：== 改为 in
        df_selection = df.query(
            "城市 in @city & 顾客类型 in @customer_type & 性别 in @gender"
        )
    return df_selection

def product_line_chart(df):
    sales_by_product_line = (
        df.groupby(by=["产品类型"])["总价"].sum().sort_values()
    )
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="总价",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>按产品类型划分的销售额</b>",
    )
    return fig_product_sales

def hour_chart(df):
    sales_by_hour = (
        df.groupby(by=["小时数"])["总价"].sum()
    )
    fig_hour_sales = px.bar(
        sales_by_hour,
        x=sales_by_hour.index,
        y="总价",
        title="<b>按小时数划分的销售额</b>",
    )
    return fig_hour_sales

def main_page_demo(df):
    """主界面函数"""
    st.title(':bar_chart: 销售仪表板')
    left_key_col, middle_key_col, right_key_col = st.columns(3)
    
    # 修复NaN转整数问题
    total_sales = int(df["总价"].sum()) if not df.empty else 0
    average_rating = round(df["评分"].mean(), 1) if not df.empty and df["评分"].notna().any() else 0.0
    star_rating_count = int(round(average_rating, 0)) if not pd.isna(average_rating) else 0
    star_rating_string = ":star:" * star_rating_count
    average_sale_by_transaction = round(df["总价"].mean(), 2) if not df.empty else 0.0
    
    with left_key_col:
        st.subheader("总销售额：")
        st.subheader(f"RMB ¥ {total_sales:,}")
    
    with middle_key_col:
        st.subheader("顾客评分的平均值：")
        st.subheader(f"{average_rating} {star_rating_string}")
    
    with right_key_col:
        st.subheader("每单的平均销售额：")
        st.subheader(f"RMB ¥ {average_sale_by_transaction}")
    
    st.divider()
    
    # 空数据提示
    if df.empty:
        st.warning("暂无符合筛选条件的数据，请调整筛选条件！")
        return
    
    left_chart_col, right_chart_col = st.columns(2)
    with left_chart_col:
        hour_fig = hour_chart(df)
        st.plotly_chart(hour_fig, use_container_width=True)
    
    with right_chart_col:
        product_fig = product_line_chart(df)
        st.plotly_chart(product_fig, use_container_width=True)

def run_app():
    """启动应用"""
    st.set_page_config(
        page_title="销售仪表板",
        page_icon=":bar_chart:",
        layout="wide"
    )
    sale_df = get_dataframe_from_excel()
    df_selection = add_sidebar_func(sale_df)
    main_page_demo(df_selection)

if __name__ == "__main__":
    run_app()
