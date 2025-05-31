import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(file_path):
    """
    Load and clean the retail data
    
    Parameters:
    file_path (str): Path to the Excel file
    
    Returns:
    pd.DataFrame: Cleaned dataframe ready for analysis
    """
    print("🔄 Loading data from:", file_path)
    
    # Load data
    df = pd.read_excel(file_path)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    
    # Display initial info
    print(f"📊 Initial data overview:")
    print(f"   - Rows: {df.shape[0]:,}")
    print(f"   - Columns: {df.shape[1]}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    
    # Data cleaning steps
    print("\n🧹 Starting data cleaning process...")
    
    # 1. Remove rows with missing CustomerID (essential for customer analysis)
    initial_rows = len(df)
    df_clean = df.dropna(subset=['CustomerID']).copy()
    removed_customers = initial_rows - len(df_clean)
    print(f"   - Removed {removed_customers:,} rows with missing CustomerID")
    
    # 2. Handle missing Description
    missing_desc = df_clean['Description'].isnull().sum()
    df_clean['Description'] = df_clean['Description'].fillna('Unknown Product')
    print(f"   - Filled {missing_desc:,} missing product descriptions")
    
    # 3. Convert data types
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)
    print(f"   - Converted InvoiceDate to datetime format")
    
    # 4. Create additional time features
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    print(f"   - Created time-based features (Year, Month, DayOfWeek, Hour)")
    
    # 5. Calculate total amount per transaction
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    print(f"   - Calculated TotalAmount for each transaction")
    
    # 6. Remove invalid transactions
    # Remove negative quantities and prices (returns/errors)
    before_filter = len(df_clean)
    df_clean = df_clean[
        (df_clean['Quantity'] > 0) & 
        (df_clean['UnitPrice'] > 0) &
        (df_clean['TotalAmount'] > 0)
    ]
    removed_invalid = before_filter - len(df_clean)
    print(f"   - Removed {removed_invalid:,} invalid transactions (negative values)")
    
    # 7. Remove extreme outliers using IQR method
    Q1 = df_clean['TotalAmount'].quantile(0.25)
    Q3 = df_clean['TotalAmount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    before_outlier_removal = len(df_clean)
    df_clean = df_clean[
        (df_clean['TotalAmount'] >= lower_bound) & 
        (df_clean['TotalAmount'] <= upper_bound)
    ]
    removed_outliers = before_outlier_removal - len(df_clean)
    print(f"   - Removed {removed_outliers:,} outlier transactions")
    
    # Final summary
    print(f"\n✅ Data cleaning completed!")
    print(f"   - Final dataset shape: {df_clean.shape}")
    print(f"   - Data retention rate: {len(df_clean)/initial_rows*100:.1f}%")
    print(f"   - Date range: {df_clean['InvoiceDate'].min().date()} to {df_clean['InvoiceDate'].max().date()}")
    print(f"   - Unique customers: {df_clean['CustomerID'].nunique():,}")
    print(f"   - Unique products: {df_clean['StockCode'].nunique():,}")
    
    return df_clean

def analyze_sales_trends(df):
    """
    Analyze monthly sales trends
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    
    Returns:
    pd.DataFrame: Monthly sales summary
    """
    print("\n📈 Analyzing sales trends...")
    
    # Monthly sales aggregation
    monthly_sales = df.groupby(['Year', 'Month']).agg({
        'TotalAmount': ['sum', 'count', 'mean'],
        'CustomerID': 'nunique',
        'StockCode': 'nunique'
    }).round(2)
    
    # Flatten column names
    monthly_sales.columns = [
        'Total_Revenue', 'Total_Transactions', 'Avg_Transaction_Value',
        'Unique_Customers', 'Unique_Products'
    ]
    monthly_sales = monthly_sales.reset_index()
    
    # Create readable date column
    monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + \
                                monthly_sales['Month'].astype(str).str.zfill(2)
    monthly_sales['Date'] = pd.to_datetime(monthly_sales['YearMonth'])
    
    # Calculate growth rates
    monthly_sales = monthly_sales.sort_values('Date')
    monthly_sales['Revenue_Growth'] = monthly_sales['Total_Revenue'].pct_change() * 100
    monthly_sales['Customer_Growth'] = monthly_sales['Unique_Customers'].pct_change() * 100
    
    # Summary statistics
    print(f"📊 Sales Trends Summary:")
    print(f"   - Total Revenue: ${monthly_sales['Total_Revenue'].sum():,.2f}")
    print(f"   - Average Monthly Revenue: ${monthly_sales['Total_Revenue'].mean():,.2f}")
    print(f"   - Best Month: {monthly_sales.loc[monthly_sales['Total_Revenue'].idxmax(), 'YearMonth']}")
    print(f"   - Peak Revenue: ${monthly_sales['Total_Revenue'].max():,.2f}")
    print(f"   - Average Monthly Growth: {monthly_sales['Revenue_Growth'].mean():.1f}%")
    
    return monthly_sales

def plot_top_products(df, n=10):
    """
    Plot top N products by revenue with comprehensive analysis
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    n (int): Number of top products to display
    """
    print(f"\n🏆 Analyzing top {n} products...")
    
    # Aggregate product data
    product_analysis = df.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'UnitPrice': 'mean'
    }).round(2)
    
    product_analysis.columns = [
        'Total_Quantity_Sold', 'Total_Revenue', 'Number_of_Orders', 
        'Unique_Customers', 'Avg_Unit_Price'
    ]
    product_analysis = product_analysis.reset_index()
    
    # Calculate additional metrics
    product_analysis['Revenue_per_Customer'] = (
        product_analysis['Total_Revenue'] / product_analysis['Unique_Customers']
    ).round(2)
    product_analysis['Avg_Quantity_per_Order'] = (
        product_analysis['Total_Quantity_Sold'] / product_analysis['Number_of_Orders']
    ).round(2)
    
    # Get top products by revenue
    top_products = product_analysis.nlargest(n, 'Total_Revenue')
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Top {n} Products Analysis', fontsize=16, fontweight='bold')
    
    # 1. Revenue Bar Chart
    axes[0, 0].barh(range(len(top_products)), top_products['Total_Revenue'])
    axes[0, 0].set_yticks(range(len(top_products)))
    axes[0, 0].set_yticklabels([desc[:30] + '...' if len(desc) > 30 else desc 
                               for desc in top_products['Description']], fontsize=9)
    axes[0, 0].set_xlabel('Total Revenue ($)')
    axes[0, 0].set_title('Top Products by Revenue')
    axes[0, 0].invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_products['Total_Revenue']):
        axes[0, 0].text(v + max(top_products['Total_Revenue']) * 0.01, i, f'${v:,.0f}', 
                       va='center', fontweight='bold')
    
    # 2. Quantity Sold
    axes[0, 1].barh(range(len(top_products)), top_products['Total_Quantity_Sold'], color='orange')
    axes[0, 1].set_yticks(range(len(top_products)))
    axes[0, 1].set_yticklabels([desc[:30] + '...' if len(desc) > 30 else desc 
                               for desc in top_products['Description']], fontsize=9)
    axes[0, 1].set_xlabel('Total Quantity Sold')
    axes[0, 1].set_title('Top Products by Quantity')
    axes[0, 1].invert_yaxis()
    
    # 3. Customer Reach
    axes[1, 0].barh(range(len(top_products)), top_products['Unique_Customers'], color='green')
    axes[1, 0].set_yticks(range(len(top_products)))
    axes[1, 0].set_yticklabels([desc[:30] + '...' if len(desc) > 30 else desc 
                               for desc in top_products['Description']], fontsize=9)
    axes[1, 0].set_xlabel('Number of Unique Customers')
    axes[1, 0].set_title('Top Products by Customer Reach')
    axes[1, 0].invert_yaxis()
    
    # 4. Revenue vs Quantity Scatter
    axes[1, 1].scatter(top_products['Total_Quantity_Sold'], top_products['Total_Revenue'], 
                      s=100, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Total Quantity Sold')
    axes[1, 1].set_ylabel('Total Revenue ($)')
    axes[1, 1].set_title('Revenue vs Quantity Relationship')
    
    # Add product labels to scatter plot
    for i, row in top_products.iterrows():
        axes[1, 1].annotate(row['StockCode'], 
                           (row['Total_Quantity_Sold'], row['Total_Revenue']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\n📋 Top {n} Products Details:")
    print("=" * 80)
    for i, (_, product) in enumerate(top_products.iterrows(), 1):
        print(f"{i:2d}. {product['Description'][:50]}")
        print(f"    Stock Code: {product['StockCode']}")
        print(f"    Revenue: ${product['Total_Revenue']:,.2f}")
        print(f"    Quantity Sold: {product['Total_Quantity_Sold']:,}")
        print(f"    Customers: {product['Unique_Customers']:,}")
        print(f"    Avg Price: ${product['Avg_Unit_Price']:.2f}")
        print("-" * 60)
    
    return top_products

def analyze_customer_behavior(df):
    """
    Analyze customer purchasing behavior
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    
    Returns:
    pd.DataFrame: Customer analysis summary
    """
    print("\n👥 Analyzing customer behavior...")
    
    # Customer aggregation
    customer_stats = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # Number of orders
        'TotalAmount': ['sum', 'mean'],  # Total and average spending
        'Quantity': 'sum',  # Total items bought
        'InvoiceDate': ['min', 'max'],  # First and last purchase
        'StockCode': 'nunique'  # Unique products purchased
    }).round(2)
    
    # Flatten column names
    customer_stats.columns = [
        'Total_Orders', 'Total_Spent', 'Avg_Order_Value', 
        'Total_Items', 'First_Purchase', 'Last_Purchase', 'Unique_Products'
    ]
    customer_stats = customer_stats.reset_index()
    
    # Calculate customer lifetime (in days)
    customer_stats['Customer_Lifetime_Days'] = (
        customer_stats['Last_Purchase'] - customer_stats['First_Purchase']
    ).dt.days
    
    # Customer segmentation
    print(f"📊 Customer Behavior Summary:")
    print(f"   - Total Customers: {len(customer_stats):,}")
    print(f"   - Average Orders per Customer: {customer_stats['Total_Orders'].mean():.1f}")
    print(f"   - Average Spend per Customer: ${customer_stats['Total_Spent'].mean():.2f}")
    print(f"   - Top 10% customers contribute: {(customer_stats.nlargest(int(len(customer_stats)*0.1), 'Total_Spent')['Total_Spent'].sum() / customer_stats['Total_Spent'].sum() * 100):.1f}% of revenue")
    
    return customer_stats

def generate_comprehensive_report(df):
    """
    Generate a comprehensive analysis report
    
    Parameters:
    df (pd.DataFrame): Cleaned dataframe
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE BUSINESS ANALYSIS REPORT")
    print("="*80)
    
    # Overall business metrics
    total_revenue = df['TotalAmount'].sum()
    total_transactions = len(df)
    unique_customers = df['CustomerID'].nunique()
    unique_products = df['StockCode'].nunique()
    date_range = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
    
    print(f"\n🎯 KEY BUSINESS METRICS:")
    print(f"   Total Revenue: ${total_revenue:,.2f}")
    print(f"   Total Transactions: {total_transactions:,}")
    print(f"   Unique Customers: {unique_customers:,}")
    print(f"   Unique Products: {unique_products:,}")
    print(f"   Analysis Period: {date_range} days")
    print(f"   Average Daily Revenue: ${total_revenue/date_range:,.2f}")
    print(f"   Revenue per Customer: ${total_revenue/unique_customers:.2f}")
    print(f"   Average Transaction Value: ${df['TotalAmount'].mean():.2f}")
    
    # Peak performance analysis
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalAmount'].sum()
    best_day = daily_sales.idxmax()
    best_day_sales = daily_sales.max()
    
    hourly_sales = df.groupby('Hour')['TotalAmount'].sum()
    peak_hour = hourly_sales.idxmax()
    
    dow_sales = df.groupby('DayOfWeek')['TotalAmount'].sum()
    best_dow = dow_sales.idxmax()
    
    print(f"\n⏰ PEAK PERFORMANCE:")
    print(f"   Best Sales Day: {best_day} (${best_day_sales:,.2f})")
    print(f"   Peak Hour: {peak_hour}:00 (${hourly_sales[peak_hour]:,.2f})")
    print(f"   Best Day of Week: {best_dow} (${dow_sales[best_dow]:,.2f})")
    
    print("\n" + "="*80)

# Main execution function
def main(file_path):
    """
    Main function to execute the complete analysis pipeline
    
    Parameters:
    file_path (str): Path to the data file
    """
    try:
        # Step 1: Load and clean data
        df_clean = load_and_clean_data(file_path)
        
        # Step 2: Analyze sales trends
        monthly_trends = analyze_sales_trends(df_clean)
        
        # Step 3: Plot top products
        top_products = plot_top_products(df_clean, n=10)
        
        # Step 4: Analyze customer behavior
        customer_analysis = analyze_customer_behavior(df_clean)
        
        # Step 5: Generate comprehensive report
        generate_comprehensive_report(df_clean)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved and visualizations displayed.")
        
        return {
            'cleaned_data': df_clean,
            'monthly_trends': monthly_trends,
            'top_products': top_products,
            'customer_analysis': customer_analysis
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Run the complete analysis
    file_path = 'data/online_retail.xlsx'  # Update with your file path
    results = main(file_path)
    
    if results:
        print(f"\n✅ All analysis functions executed successfully!")
        print(f"🔍 Use the returned results dictionary to access:")
        print(f"   - results['cleaned_data']: Cleaned dataset")
        print(f"   - results['monthly_trends']: Monthly sales analysis")
        print(f"   - results['top_products']: Top products analysis")
        print(f"   - results['customer_analysis']: Customer behavior data")
