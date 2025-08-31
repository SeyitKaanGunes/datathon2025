import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import shap
import warnings
from scipy import stats
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# --- Veri Setlerini Yükleme ---
try:
    train_df = pd.read_csv('/kaggle/input/datathon/train.csv')
    test_df = pd.read_csv('/kaggle/input/datathon/test.csv')
    print("Veri setleri başarıyla yüklendi.")
    print(f"Train verisi başlangıç boyutu: {train_df.shape}")
    print(f"Test verisi başlangıç boyutu: {test_df.shape}")
except FileNotFoundError:
    print("Hata: 'train.csv' ve 'test.csv' dosyaları bulunamadı.")
    exit()

print("-" * 40)

# --- Anomali Tespiti ---
train_session_user_counts = train_df.groupby('user_session')['user_id'].nunique()
train_anomalous_sessions = train_session_user_counts[train_session_user_counts > 1].index.tolist()

test_session_user_counts = test_df.groupby('user_session')['user_id'].nunique()
test_anomalous_sessions = test_session_user_counts[test_session_user_counts > 1].index.tolist()

all_anomalous_sessions = set(train_anomalous_sessions + test_anomalous_sessions)
print(f"Toplam {len(all_anomalous_sessions)} benzersiz anomalili oturum tespit edildi: {list(all_anomalous_sessions)}")

print("-" * 40)

# --- Veri Setlerini Ayırma (Silme Yerine) ---
train_clean_df = train_df[~train_df['user_session'].isin(all_anomalous_sessions)].copy()
print(f"Train verisi temizlendi. Yeni boyut: {train_clean_df.shape}")

test_clean_df = test_df[~test_df['user_session'].isin(all_anomalous_sessions)].copy()
test_anomalous_df = test_df[test_df['user_session'].isin(all_anomalous_sessions)].copy()

print(f"Test verisi ikiye ayrıldı:")
print(f"  - Tahmin edilecek temiz test verisi boyutu: {test_clean_df.shape}")
print(f"  - Sonradan eklenecek anomalili test verisi boyutu: {test_anomalous_df.shape}")

print("-" * 40)

# --- Anomalili Oturumlar İçin Atanacak Değeri Belirleme ---
placeholder_value = train_clean_df['session_value'].median()
print(f"Anomalili oturumlara atanacak 'placeholder' değer (medyan): {placeholder_value:.4f}")

anomalous_session_ids_in_test = test_anomalous_df['user_session'].unique()
anomalous_predictions = pd.DataFrame({
    'user_session': anomalous_session_ids_in_test,
    'session_value': placeholder_value
})

print(f"\nSubmission için {anomalous_predictions.shape[0]} adet anomalili oturumun tahmini hazırlandı.")
print("Bu tahminler, modelin temiz veri üzerindeki tahminleriyle en sonda birleştirilecek.")

# --- Ortak Oturumları Tespit Etme ---
train_clean_sessions = set(train_clean_df['user_session'].unique())
test_clean_sessions = set(test_clean_df['user_session'].unique())
common_sessions = train_clean_sessions.intersection(test_clean_sessions)

print(f"Temizlenmiş Train ve Test setleri arasında {len(common_sessions)} adet ortak oturum bulundu.")
print("-" * 40)

# --- Ortak Oturumlar İçin "Cevap Anahtarı" Oluşturma ---
common_sessions_data = train_clean_df[train_clean_df['user_session'].isin(common_sessions)]
common_session_answers = common_sessions_data[['user_session', 'session_value']].drop_duplicates().reset_index(drop=True)

print(f"Ortak {common_session_answers.shape[0]} oturum için 'cevap anahtarı' oluşturuldu.")
print("Bu cevaplar, model tahminleriyle en sonda birleştirilecek.")
print("Örnek Cevaplar:")
print(common_session_answers.head())

print("-" * 40)

# --- Modelleme İçin Nihai Veri Setlerini Oluşturma ---
train_model_df = train_clean_df[~train_clean_df['user_session'].isin(common_sessions)].copy()
test_model_df = test_clean_df[~test_clean_df['user_session'].isin(common_sessions)].copy()

print("Modelleme için nihai veri setleri oluşturuldu:")
print(f"  - Model Eğitimi Verisi Boyutu (train_model_df): {train_model_df.shape}")
print(f"  - Model Tahmin Verisi Boyutu (test_model_df): {test_model_df.shape}")

# --- Geliştirilmiş Özellik Mühendisliği Fonksiyonu ---

def create_enhanced_session_features(df):
    """
    Olay bazlı veriden oturum bazlı özellikler türeten geliştirilmiş fonksiyon.
    String özellikleri hariç tutarak sadece numerik özellikler üretir.
    """
    print("Gelişmiş feature engineering başlatılıyor...")
    
    # event_time sütununu datetime formatına çevirelim
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Zaman bazlı ek özellikler ekleyelim
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['month'] = df['event_time'].dt.month
    df['day'] = df['event_time'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Event sıra numarası (her session içinde)
    df['event_order'] = df.groupby('user_session').cumcount() + 1
    
    # --- 1. Temel Aggregasyonlar ---
    aggregations = {
        'event_time': ['min', 'max', 'count'],
        'product_id': ['nunique', 'count'],
        'category_id': ['nunique', 'count'],
        'event_type': ['count'],
        'user_id': ['nunique'],
        'hour': ['min', 'max', 'mean', 'std'],
        'day_of_week': ['min', 'max', 'mean'],
        'month': ['nunique'],
        'day': ['nunique'],
        'event_order': ['max']  # Son event sıra numarası = toplam event sayısı
    }
    
    session_df = df.groupby('user_session').agg(aggregations)
    session_df.columns = ['_'.join(col).strip() for col in session_df.columns.values]
    session_df.reset_index(inplace=True)

    print("Temel aggregasyonlar tamamlandı.")

    # --- 2. Event Type Özelikleri ---
    event_type_counts = pd.crosstab(df['user_session'], df['event_type'])
    session_df = session_df.merge(event_type_counts, on='user_session', how='left')
    session_df[event_type_counts.columns] = session_df[event_type_counts.columns].fillna(0)

    # --- 3. Ürün ve Kategori Bazlı Gelişmiş Özellikler (String özellikleri hariç) ---
    
    # Her session'daki ürün çeşitliliği (product_mode string olduğu için hariç)
    product_stats = df.groupby('user_session')['product_id'].agg([
        ('product_unique_ratio', lambda x: len(x.unique()) / len(x)),
        ('most_viewed_product_count', lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0),
        ('most_viewed_product_ratio', lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0)
    ])
    session_df = session_df.merge(product_stats, on='user_session', how='left')

    # Kategori çeşitliliği
    category_stats = df.groupby('user_session')['category_id'].agg([
        ('category_unique_ratio', lambda x: len(x.unique()) / len(x)),
        ('most_viewed_category_count', lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0),
        ('most_viewed_category_ratio', lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0),
        ('category_switch_count', lambda x: sum(x.iloc[i] != x.iloc[i-1] for i in range(1, len(x))) if len(x) > 1 else 0)
    ])
    session_df = session_df.merge(category_stats, on='user_session', how='left')

    print("Ürün ve kategori özellikleri tamamlandı.")

    # --- 4. Zaman Bazlı Gelişmiş Özellikler ---
    
    # Session süreleri
    session_df['session_duration_seconds'] = (session_df['event_time_max'] - session_df['event_time_min']).dt.total_seconds()
    session_df['session_duration_minutes'] = session_df['session_duration_seconds'] / 60
    session_df['session_duration_hours'] = session_df['session_duration_seconds'] / 3600
    
    # Zaman dilimleri
    session_df['session_start_hour'] = session_df['hour_min']
    session_df['session_end_hour'] = session_df['hour_max']
    session_df['session_hour_span'] = session_df['hour_max'] - session_df['hour_min']
    session_df['avg_session_hour'] = session_df['hour_mean']
    session_df['hour_std'] = session_df['hour_std'].fillna(0)
    
    # Zaman bazlı kategoriler
    session_df['is_morning'] = ((session_df['session_start_hour'] >= 6) & (session_df['session_start_hour'] < 12)).astype(int)
    session_df['is_afternoon'] = ((session_df['session_start_hour'] >= 12) & (session_df['session_start_hour'] < 18)).astype(int)
    session_df['is_evening'] = ((session_df['session_start_hour'] >= 18) & (session_df['session_start_hour'] < 22)).astype(int)
    session_df['is_night'] = ((session_df['session_start_hour'] >= 22) | (session_df['session_start_hour'] < 6)).astype(int)
    
    # Hafta içi/sonu özellikleri
    session_df['starts_weekend'] = (session_df['day_of_week_min'] >= 5).astype(int)
    session_df['ends_weekend'] = (session_df['day_of_week_max'] >= 5).astype(int)
    session_df['crosses_weekend'] = ((session_df['day_of_week_min'] < 5) & (session_df['day_of_week_max'] >= 5)).astype(int)
    
    print("Zaman bazlı özellikler tamamlandı.")

    # --- 5. Davranışsal Özellikler (Optimized) ---
    
    print("  Sequence analizi başlıyor...")
    
    # Daha hızlı sequence analizi için vectorized operations
    session_events = df.sort_values(['user_session', 'event_time']).groupby('user_session')
    
    # İlk ve son event'ler
    first_events = session_events['event_type'].first().reset_index()
    last_events = session_events['event_type'].last().reset_index()
    last_events = last_events.rename(columns={'event_type': 'last_event_type'})
    
    # Event type varlık kontrolü (daha hızlı)
    event_presence = df.groupby('user_session')['event_type'].apply(lambda x: set(x.values)).reset_index()
    event_presence.columns = ['user_session', 'event_set']
    
    # Sequence features'ları oluştur
    sequence_features = first_events.merge(last_events, on='user_session').merge(event_presence, on='user_session')
    
    # Feature'ları hesapla
    sequence_features['first_event_is_view'] = (sequence_features['event_type'] == 'VIEW').astype(int)
    sequence_features['first_event_is_cart'] = (sequence_features['event_type'] == 'ADD_CART').astype(int)
    sequence_features['first_event_is_buy'] = (sequence_features['event_type'] == 'BUY').astype(int)
    sequence_features['last_event_is_view'] = (sequence_features['last_event_type'] == 'VIEW').astype(int)
    sequence_features['last_event_is_buy'] = (sequence_features['last_event_type'] == 'BUY').astype(int)
    sequence_features['last_event_is_cart'] = (sequence_features['last_event_type'] == 'ADD_CART').astype(int)
    sequence_features['view_to_buy_direct'] = ((sequence_features['event_type'] == 'VIEW') & 
                                              (sequence_features['last_event_type'] == 'BUY')).astype(int)
    sequence_features['view_cart_buy_funnel'] = sequence_features['event_set'].apply(
        lambda x: 1 if {'VIEW', 'ADD_CART', 'BUY'}.issubset(x) else 0)
    sequence_features['has_view_and_buy'] = sequence_features['event_set'].apply(
        lambda x: 1 if {'VIEW', 'BUY'}.issubset(x) else 0)
    sequence_features['has_cart_and_buy'] = sequence_features['event_set'].apply(
        lambda x: 1 if {'ADD_CART', 'BUY'}.issubset(x) else 0)
    sequence_features['event_type_diversity'] = sequence_features['event_set'].apply(len)
    
    # Gereksiz sütunları kaldır
    sequence_features = sequence_features.drop(['event_type', 'last_event_type', 'event_set'], axis=1)
    
    session_df = session_df.merge(sequence_features, on='user_session', how='left')
    print("  Sequence analizi tamamlandı.")

    print("Davranışsal özellikler tamamlandı.")

    # --- 6. İstatistiksel Özellikler (Optimized) ---
    
    # Event timing analizi - daha hızlı batch processing
    print("  Timing analizi başlıyor...")
    
    # Grupları önceden hesapla
    grouped = df.sort_values('event_time').groupby('user_session')
    
    # Vectorized timing hesaplamaları
    timing_stats = []
    for name, group in grouped:
        if len(group) > 1:
            time_diffs = group['event_time'].diff().dt.total_seconds().dropna()
            if len(time_diffs) > 0:
                q75 = time_diffs.quantile(0.75)
                timing_stats.append({
                    'user_session': name,
                    'avg_time_between_events': time_diffs.mean(),
                    'std_time_between_events': time_diffs.std(),
                    'min_time_between_events': time_diffs.min(),
                    'max_time_between_events': time_diffs.max(),
                    'total_pause_time': time_diffs.sum(),
                    'long_pauses_count': (time_diffs > q75).sum()
                })
            else:
                timing_stats.append({
                    'user_session': name,
                    'avg_time_between_events': 0,
                    'std_time_between_events': 0,
                    'min_time_between_events': 0,
                    'max_time_between_events': 0,
                    'total_pause_time': 0,
                    'long_pauses_count': 0
                })
        else:
            timing_stats.append({
                'user_session': name,
                'avg_time_between_events': 0,
                'std_time_between_events': 0,
                'min_time_between_events': 0,
                'max_time_between_events': 0,
                'total_pause_time': 0,
                'long_pauses_count': 0
            })
    
    timing_df = pd.DataFrame(timing_stats)
    session_df = session_df.merge(timing_df, on='user_session', how='left')
    print("  Timing analizi tamamlandı.")

    # --- 7. Event Type Oranları ve İlişkileri ---
    total_events = session_df['event_type_count']
    
    for col in event_type_counts.columns:
        session_df[f'{col}_ratio'] = session_df[col] / (total_events + 1e-8)
        session_df[f'{col}_per_minute'] = session_df[col] / (session_df['session_duration_minutes'] + 1e-8)
        session_df[f'{col}_per_product'] = session_df[col] / (session_df['product_id_nunique'] + 1e-8)
    
    # Cross-ratios (çapraz oranlar)
    if 'VIEW' in session_df.columns and 'ADD_CART' in session_df.columns:
        session_df['cart_conversion_rate'] = session_df['ADD_CART'] / (session_df['VIEW'] + 1e-8)
    if 'BUY' in session_df.columns and 'ADD_CART' in session_df.columns:
        session_df['purchase_conversion_rate'] = session_df['BUY'] / (session_df['ADD_CART'] + 1e-8)
    if 'BUY' in session_df.columns and 'VIEW' in session_df.columns:
        session_df['direct_purchase_rate'] = session_df['BUY'] / (session_df['VIEW'] + 1e-8)
    if 'REMOVE_CART' in session_df.columns and 'ADD_CART' in session_df.columns:
        session_df['cart_abandonment_rate'] = session_df['REMOVE_CART'] / (session_df['ADD_CART'] + 1e-8)

    print("Event type oranları tamamlandı.")

    # --- 8. Aktivite Yoğunluğu ve Engagement Metrikleri ---
    
    session_df['events_per_minute'] = session_df['event_type_count'] / (session_df['session_duration_minutes'] + 1e-8)
    session_df['events_per_hour'] = session_df['event_type_count'] / (session_df['session_duration_hours'] + 1e-8)
    session_df['products_per_minute'] = session_df['product_id_nunique'] / (session_df['session_duration_minutes'] + 1e-8)
    session_df['categories_per_minute'] = session_df['category_id_nunique'] / (session_df['session_duration_minutes'] + 1e-8)
    
    # Session büyüklük kategorileri
    session_df['is_short_session'] = (session_df['session_duration_minutes'] < 5).astype(int)
    session_df['is_medium_session'] = ((session_df['session_duration_minutes'] >= 5) & (session_df['session_duration_minutes'] <= 30)).astype(int)
    session_df['is_long_session'] = (session_df['session_duration_minutes'] > 30).astype(int)
    session_df['is_very_long_session'] = (session_df['session_duration_minutes'] > 60).astype(int)
    
    # Aktivite seviyeleri
    event_q75 = session_df['event_type_count'].quantile(0.75)
    event_q25 = session_df['event_type_count'].quantile(0.25)
    
    session_df['is_high_activity'] = (session_df['event_type_count'] > event_q75).astype(int)
    session_df['is_low_activity'] = (session_df['event_type_count'] < event_q25).astype(int)
    session_df['is_very_high_activity'] = (session_df['event_type_count'] > session_df['event_type_count'].quantile(0.9)).astype(int)

    print("Aktivite metrikleri tamamlandı.")

    # --- 9. Kompleks Özellikler ve İndeksler ---
    
    # Çeşitlilik indeksleri
    session_df['product_diversity_index'] = session_df['product_id_nunique'] / (session_df['event_type_count'] + 1e-8)
    session_df['category_diversity_index'] = session_df['category_id_nunique'] / (session_df['event_type_count'] + 1e-8)
    session_df['exploration_index'] = (session_df['product_id_nunique'] * session_df['category_id_nunique']) / (session_df['event_type_count'] + 1e-8)
    
    # Verimlilik metrikleri
    session_df['session_efficiency'] = session_df['product_id_nunique'] / (session_df['session_duration_minutes'] + 1e-8)
    session_df['browsing_intensity'] = session_df['event_type_count'] / (session_df['product_id_nunique'] + 1e-8)
    session_df['focus_score'] = session_df['most_viewed_product_ratio'] * session_df['most_viewed_category_ratio']
    
    # Engagement skorları
    if 'BUY' in session_df.columns:
        session_df['purchase_intent_score'] = (session_df['BUY'] + session_df.get('ADD_CART', 0)) / (session_df['VIEW'] + 1e-8)
    
    session_df['exploration_breadth'] = session_df['product_id_nunique'] + session_df['category_id_nunique']
    session_df['exploration_depth'] = session_df['event_type_count'] / (session_df['exploration_breadth'] + 1e-8)

    print("Kompleks özellikler tamamlandı.")

    # --- 10. Matematiksel Dönüşümler ---
    
    # Log transformasyonları (pozitif değerler için)
    numerical_cols = ['event_type_count', 'session_duration_minutes', 'product_id_nunique', 'category_id_nunique']
    for col in numerical_cols:
        if col in session_df.columns:
            session_df[f'{col}_log'] = np.log1p(session_df[col])
            session_df[f'{col}_sqrt'] = np.sqrt(session_df[col])
    
    # Polynomial features (seçili önemli özellikler için)
    important_cols = ['event_type_count', 'session_duration_minutes', 'product_id_nunique']
    for col in important_cols:
        if col in session_df.columns:
            session_df[f'{col}_squared'] = session_df[col] ** 2
    
    print("Matematiksel dönüşümler tamamlandı.")

    # --- 11. Binning Features ---
    
    # Event count bins
    session_df['event_count_bin'] = pd.cut(session_df['event_type_count'], 
                                         bins=5, labels=False)
    
    # Duration bins
    session_df['duration_bin'] = pd.cut(session_df['session_duration_minutes'], 
                                      bins=5, labels=False)
    
    # Hour bins
    session_df['hour_bin'] = pd.cut(session_df['session_start_hour'], 
                                  bins=[0, 6, 12, 18, 24], labels=False)

    print("Binning işlemleri tamamlandı.")

    # --- 12. Interaction Features ---
    
    # Bazı önemli etkileşim özellikleri
    session_df['duration_x_events'] = session_df['session_duration_minutes'] * session_df['event_type_count']
    session_df['products_x_categories'] = session_df['product_id_nunique'] * session_df['category_id_nunique']
    session_df['hour_x_duration'] = session_df['session_start_hour'] * session_df['session_duration_minutes']
    
    if 'VIEW' in session_df.columns and 'BUY' in session_df.columns:
        session_df['view_buy_interaction'] = session_df['VIEW'] * session_df['BUY']
    
    print("Etkileşim özellikleri tamamlandı.")

    # --- Temizlik ---
    # Artık ihtiyacımız olmayan sütunları kaldıralım
    columns_to_drop = ['event_time_min', 'event_time_max', 'event_time_count']
    session_df = session_df.drop(columns=[col for col in columns_to_drop if col in session_df.columns])
    
    # Infinity ve NaN değerlerini temizle
    session_df = session_df.replace([np.inf, -np.inf], np.nan)
    session_df = session_df.fillna(0)
    
    print(f"Özellik mühendisliği tamamlandı. Toplam {session_df.shape[1]} özellik oluşturuldu.")
    
    return session_df

# --- Fonksiyonu Çalıştırma ---

print("Train verisi için gelişmiş özellikler oluşturuluyor...")
train_features_df = create_enhanced_session_features(train_model_df)

print("Test verisi için gelişmiş özellikler oluşturuluyor...")
test_features_df = create_enhanced_session_features(test_model_df)

# --- Hedef Değişkeni (session_value) Geri Ekleme ---
session_values = train_model_df[['user_session', 'session_value']].drop_duplicates()
train_features_df = train_features_df.merge(session_values, on='user_session', how='left')

print("\nGelişmiş özellik mühendisliği tamamlandı.")
print(f"Train features boyutu: {train_features_df.shape}")
print(f"Test features boyutu: {test_features_df.shape}")

print("\nOluşturulan Train Features Örneği:")
print(train_features_df.head())

# =====================================
# SHAP FEATURE SELECTION EKLEME
# =====================================

print("\n" + "="*60)
print("SHAP FEATURE SELECTION BAŞLANIYOR")
print("="*60)

# --- Veriyi Hazırlama ---
target = 'session_value'
features_to_drop = ['user_session', target]

X_train_full = train_features_df.drop(columns=features_to_drop)
y_train = train_features_df[target].values
X_test_full = test_features_df.drop(columns=['user_session'])

# String olan sütunları tespit edip çıkaralım
print("String sütunlar kontrol ediliyor...")
string_columns = []
for col in X_train_full.columns:
    if X_train_full[col].dtype == 'object':
        string_columns.append(col)
        print(f"  String sütun bulundu ve çıkarılıyor: {col}")

# String sütunları çıkar
if string_columns:
    X_train_full = X_train_full.drop(columns=string_columns)
    X_test_full = X_test_full.drop(columns=string_columns, errors='ignore')
    print(f"Toplam {len(string_columns)} string sütun çıkarıldı.")

# Kalan NaN değerleri kontrol et ve temizle
X_train_full = X_train_full.fillna(0)
X_test_full = X_test_full.fillna(0)

# Infinity değerlerini temizle
X_train_full = X_train_full.replace([np.inf, -np.inf], 0)
X_test_full = X_test_full.replace([np.inf, -np.inf], 0)

print(f"Toplam özellik sayısı (string sütunlar çıkarıldıktan sonra): {X_train_full.shape[1]}")

# --- SHAP için Hızlı Random Forest Modeli Eğitimi ---
print("SHAP analizi için Random Forest modeli eğitiliyor...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Veri boyutu çok büyükse, SHAP analizi için sample alıyoruz
sample_size = min(5000, X_train_full.shape[0])
if X_train_full.shape[0] > sample_size:
    sample_idx = np.random.choice(X_train_full.shape[0], sample_size, replace=False)
    X_train_sample = X_train_full.iloc[sample_idx]
    y_train_sample = y_train[sample_idx]
else:
    X_train_sample = X_train_full
    y_train_sample = y_train

print(f"SHAP analizi için kullanılan sample boyutu: {X_train_sample.shape}")

rf_model.fit(X_train_sample, y_train_sample)

# --- SHAP Explainer Oluşturma ve Values Hesaplama ---
print("SHAP values hesaplanıyor...")

explainer = shap.TreeExplainer(rf_model)
background_size = min(100, X_train_sample.shape[0])
background_data = X_train_sample.iloc[:background_size]
shap_values = explainer.shap_values(background_data)

print(f"SHAP values hesaplandı. Boyut: {shap_values.shape}")

# --- Feature Importance'ları SHAP'tan Al ---
feature_importance_shap = np.mean(np.abs(shap_values), axis=0)

shap_importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'shap_importance': feature_importance_shap
}).sort_values('shap_importance', ascending=False)

print("\nSHAP Feature Importance (Top 30):")
print(shap_importance_df.head(30))

# --- Feature Selection ---
TOP_N_FEATURES = 25  # Daha fazla feature seçelim

selected_features = shap_importance_df.head(TOP_N_FEATURES)['feature'].tolist()
print(f"\nSeçilen {TOP_N_FEATURES} feature:")
for i, feat in enumerate(selected_features, 1):
    importance = shap_importance_df[shap_importance_df['feature'] == feat]['shap_importance'].iloc[0]
    print(f"{i:2d}. {feat:45s}: {importance:.6f}")

# --- Seçilen Feature'larla Veriyi Filtrele ---
X_train_selected = X_train_full[selected_features]
X_test_selected = X_test_full[selected_features]

print(f"\nFeature selection tamamlandı:")
print(f"Orijinal feature sayısı: {X_train_full.shape[1]}")
print(f"Seçilen feature sayısı: {X_train_selected.shape[1]}")
print(f"Feature reduction oranı: {(1 - X_train_selected.shape[1]/X_train_full.shape[1])*100:.1f}%")

# =====================================
# TABNET MODELİNİ SEÇİLEN FEATURE'LARLA EĞİTME
# =====================================

print("\n" + "="*60)
print("TABNET MODELİ SEÇİLEN FEATURE'LARLA EĞİTİLİYOR")
print("="*60)

# Normalize etme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

print("Model eğitimi için veri hazırlandı.")
print(f"Eğitim için kullanılacak özellik (feature) sayısı: {X_train_scaled.shape[1]}")
print(f"Eğitim veri boyutu: {X_train_scaled.shape}")
print(f"Test veri boyutu: {X_test_scaled.shape}")

print("-" * 40)

# --- 5-Fold Çapraz Doğrulama ile TabNet Eğitimi ---

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(X_train_scaled.shape[0])
test_predictions = np.zeros(X_test_scaled.shape[0])
oof_rmse_scores = []
all_feature_importances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    print(f"===== Fold {fold+1}/{N_SPLITS} Başladı =====")
    
    # Veriyi bu fold için ayır
    X_train_fold, y_train_fold = X_train_scaled[train_idx], y_train[train_idx]
    X_val_fold, y_val_fold = X_train_scaled[val_idx], y_train[val_idx]
    
    print(f"Train boyutu: {X_train_fold.shape[0]}, Val boyutu: {X_val_fold.shape[0]}")
    
    # TabNet Modelini Tanımla
    tabnet_model = TabNetRegressor(
        n_d=64,                    # Decision step'teki boyut
        n_a=64,                    # Attention'daki boyut  
        n_steps=6,                 # Karar adımı sayısı (artırdık)
        gamma=1.5,                 # Feature reusage için gamma (artırdık)
        n_independent=2,           # Independent GLU layers
        n_shared=2,                # Shared GLU layers
        lambda_sparse=0.0001,      # Sparsity regularization
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=0.02, weight_decay=0.0001),
        mask_type='entmax',        # Attention mask type
        scheduler_params=dict(
            mode="min", 
            patience=15, 
            min_lr=1e-5, 
            factor=0.5
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=1,
        seed=42
    )
    
    # Modeli Eğit
    tabnet_model.fit(
        X_train_fold, y_train_fold.reshape(-1, 1),
        eval_set=[(X_val_fold, y_val_fold.reshape(-1, 1))],
        eval_name=['val'],
        eval_metric=['mse'],
        max_epochs=250,
        patience=25,
        batch_size=2048,
        virtual_batch_size=512,
        num_workers=0,
        drop_last=False
    )
    
    # Tahminleri Yap
    val_preds = tabnet_model.predict(X_val_fold).flatten()
    oof_predictions[val_idx] = val_preds
    
    fold_test_preds = tabnet_model.predict(X_test_scaled).flatten()
    test_predictions += fold_test_preds / N_SPLITS
    
    # Skoru Hesapla ve Kaydet
    rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
    oof_rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")
    
    # Feature importance'ı kaydet
    feature_importances = tabnet_model.feature_importances_
    all_feature_importances.append(feature_importances)

print("-" * 40)
print(f"Ortalama Cross-Validation RMSE: {np.mean(oof_rmse_scores):.4f} (± {np.std(oof_rmse_scores):.4f})")

# --- OOF Skoru Hesaplama ---
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
oof_mse = mean_squared_error(y_train, oof_predictions)
print(f"Genel OOF RMSE (Tüm Veri): {oof_rmse:.4f}")
print(f"Genel OOF MSE (Tüm Veri): {oof_mse:.4f}")

# --- Ortalama Feature Importance Hesaplama ---
mean_feature_importances = np.mean(all_feature_importances, axis=0)
feature_names = selected_features

# Tüm featureların skorunu göster
importance_df = pd.DataFrame({
    'feature': feature_names,
    'tabnet_importance': mean_feature_importances
}).sort_values('tabnet_importance', ascending=False)

print(f"\n===== SEÇILMIŞ FEATURELARIN TABNET ÖNEM SKORLARI =====")
for idx, row in importance_df.iterrows():
    print(f"{row['feature']:50s}: {row['tabnet_importance']:.6f}")

print(f"\nEn Önemli 20 Özellik (TabNet):")
print(importance_df.head(20))

# --- Tahminleri DataFrame'e Dönüştürme ---
model_predictions_df = pd.DataFrame({
    'user_session': test_features_df['user_session'],
    'session_value': test_predictions
})

print("\nTabNet modelinin temiz veri üzerindeki tahminleri tamamlandı.")
print(model_predictions_df.head())

# --- Model Karşılaştırması İçin İstatistikler ---
print(f"\nTahmin İstatistikleri:")
print(f"Min tahmin: {test_predictions.min():.4f}")
print(f"Max tahmin: {test_predictions.max():.4f}")
print(f"Ortalama tahmin: {test_predictions.mean():.4f}")
print(f"Medyan tahmin: {np.median(test_predictions):.4f}")
print(f"Std tahmin: {test_predictions.std():.4f}")

# =====================================
# FINAL SUBMISSION
# =====================================

print("\n" + "="*60)
print("FINAL SUBMISSION DOSYASI HAZIRLANYOR")
print("="*60)

# --- Final Submission Dosyası Oluşturma ---
print("Final submission dosyası hazırlanıyor...")
print(f"Model tahminleri: {model_predictions_df.shape[0]} satır")
print(f"Bilinen cevaplar (ortak oturumlar): {common_session_answers.shape[0]} satır")
print(f"Anomalili oturumlar (medyan ataması): {anomalous_predictions.shape[0]} satır")
print("-" * 40)

# --- Parçaları Birleştirme (Concatenate) ---
final_submission_df = pd.concat(
    [model_predictions_df, common_session_answers, anomalous_predictions],
    ignore_index=True
)

# Negatif tahminleri 0'a çekelim
final_submission_df['session_value'] = final_submission_df['session_value'].clip(lower=0)

# --- Orijinal Test Seti ile Kontrol ---
original_test_sessions = test_df['user_session'].unique()
submission_sessions = final_submission_df['user_session'].unique()

print(f"Orijinal test setindeki benzersiz oturum sayısı: {len(original_test_sessions)}")
print(f"Final submission dosyasındaki benzersiz oturum sayısı: {len(submission_sessions)}")

if len(original_test_sessions) == len(submission_sessions):
    print("\nKontrol Başarılı: Submission dosyasındaki oturum sayısı orijinal test seti ile eşleşiyor.")
else:
    print("\n!!! UYARI: Submission dosyasındaki oturum sayısı orijinal test seti ile EŞLEŞMİYOR! Gözden geçirin.")

# --- CSV Dosyasını Kaydetme ---
submission_filename = 'submission_enhanced_features_v3_fixed.csv'
final_submission_df.to_csv(submission_filename, index=False)

print(f"\n'{submission_filename}' dosyası başarıyla oluşturuldu ve kaydedildi.")
print("Bu dosyayı Kaggle'a yükleyebilirsiniz.")

# Oluşturulan dosyanın ilk 5 satırını gösterelim
print("\nOluşturulan Submission Dosyasının İlk 5 Satırı:")
print(final_submission_df.head())

print("\n" + "="*60)
print("GELİŞMİŞ FEATURE ENGINEERING İLE MODEL EĞİTİMİ TAMAMLANDI!")
print("="*60)
print(f"Toplam oluşturulan feature sayısı: {X_train_full.shape[1]}")
print(f"Seçilen feature sayısı: {len(selected_features)}")
print(f"Final OOF RMSE: {oof_rmse:.4f}")
print(f"Final OOF MSE: {oof_mse:.4f}")
print("="*60)

# --- Bonus: Feature Category Analysis ---
print("\n" + "="*40)
print("FEATURE CATEGORY ANALYSIS")
print("="*40)

# En önemli feature'ları kategorilere ayıralım
selected_features_df = importance_df.head(20).copy()

def categorize_feature(feature_name):
    if any(x in feature_name.lower() for x in ['time', 'hour', 'duration', 'minute', 'second']):
        return 'Temporal'
    elif any(x in feature_name.lower() for x in ['product', 'category']):
        return 'Product/Category'
    elif any(x in feature_name.lower() for x in ['view', 'buy', 'cart', 'event']):
        return 'Event/Behavior'
    elif any(x in feature_name.lower() for x in ['ratio', 'rate', 'conversion']):
        return 'Conversion/Ratio'
    elif any(x in feature_name.lower() for x in ['count', 'nunique', 'sum']):
        return 'Aggregation'
    elif any(x in feature_name.lower() for x in ['diversity', 'exploration', 'efficiency']):
        return 'Engagement'
    else:
        return 'Other'

selected_features_df['category'] = selected_features_df['feature'].apply(categorize_feature)

print("En Önemli 20 Feature'ın Kategorilere Göre Dağılımı:")
category_counts = selected_features_df['category'].value_counts()
for category, count in category_counts.items():
    print(f"{category:20s}: {count} features")
    features_in_category = selected_features_df[selected_features_df['category'] == category]['feature'].tolist()
    for feat in features_in_category[:3]:  # İlk 3'ünü göster
        importance = selected_features_df[selected_features_df['feature'] == feat]['tabnet_importance'].iloc[0]
        print(f"  - {feat:35s}: {importance:.6f}")
    if len(features_in_category) > 3:
        print(f"  ... ve {len(features_in_category)-3} tane daha")
    print()

print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)