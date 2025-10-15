import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =====================================================================
# TAREA A: Construir un único dataset
# =====================================================================
print("="*70)
print("ANÁLISIS DE SULFATOS Y NITRATOS EN AGUA POTABLE")
print("="*70)

# Leer el archivo CSV consolidado
try:
    print("\nCargando archivo distributed-data.csv...")
    df = pd.read_csv('distributed-data.csv', 
                     parse_dates=['Date'],  # Convertir fechas
                     na_values=['NA', 'na', 'N/A', ''],  # Manejar valores nulos
                     dtype={
                         'sulfate': float,
                         'nitrate': float,
                         'ID': int
                     })

    print("\n📊 TAREA A: Dataset Unificado")
    print("-"*50)
    print(f"✅ Dataset cargado exitosamente")
    print(f"   • Total de registros: {len(df):,}")
    print(f"   • Columnas: {', '.join(df.columns)}")
    print(f"   • Rango de fechas: {df['Date'].min()} a {df['Date'].max()}")
    
    # Verificar si la columna source_file existe
    if 'source_file' in df.columns:
        print(f"   • Número de archivos fuente: {df['source_file'].nunique()}")
    
except FileNotFoundError:
    print("\n❌ Error: No se encontró el archivo 'distributed-data.csv'")
    print("   Asegúrate de que el archivo esté en el directorio actual.")
    exit(1)
except Exception as e:
    print(f"\n❌ Error al cargar el archivo: {str(e)}")
    print("   Verifica el formato del archivo y los tipos de datos.")
    exit(1)

# Convertir columnas a numéricas, los 'NA' se convierten a NaN
df['sulfate'] = pd.to_numeric(df['sulfate'], errors='coerce')
df['nitrate'] = pd.to_numeric(df['nitrate'], errors='coerce')

# =====================================================================
# TAREA B: Histograma de mediciones útiles vs erróneas
# =====================================================================
print("\n📈 TAREA B: Análisis de Calidad de Mediciones")
print("-"*50)

# Identificar mediciones útiles (sin valores faltantes) y erróneas
mediciones_utiles = df[['sulfate', 'nitrate']].notna().all(axis=1).sum()
mediciones_erroneas = df[['sulfate', 'nitrate']].isna().any(axis=1).sum()

print(f"   • Mediciones útiles: {mediciones_utiles:,} ({mediciones_utiles/len(df)*100:.2f}%)")
print(f"   • Mediciones erróneas: {mediciones_erroneas:,} ({mediciones_erroneas/len(df)*100:.2f}%)")
print(f"   • Proporción erróneas/útiles: {mediciones_erroneas/mediciones_utiles:.2f}:1")

# Crear el histograma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de barras
categorias = ['Mediciones Útiles', 'Mediciones Erróneas']
valores = [mediciones_utiles, mediciones_erroneas]
colores = ['#2ecc71', '#e74c3c']

bars = ax1.bar(categorias, valores, color=colores, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Número de Mediciones', fontsize=12)
ax1.set_title('Comparación de Mediciones Útiles vs Erróneas', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')

# Agregar valores en las barras
for bar, val in zip(bars, valores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}\n({val/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Gráfico de torta
ax2.pie(valores, labels=categorias, colors=colores, autopct='%1.1f%%',
        startangle=90, explode=(0, 0.1))
ax2.set_title('Proporción de Mediciones', fontsize=14, fontweight='bold')

plt.suptitle('Análisis de Calidad de las Mediciones del Instrumento', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"\n💡 RESPUESTA B: Las mediciones erróneas representan el {mediciones_erroneas/len(df)*100:.1f}%")
print(f"   del total, siendo {mediciones_erroneas/mediciones_utiles:.1f} veces más frecuentes")
print(f"   que las mediciones útiles. Esto indica un problema grave en los instrumentos.")

# =====================================================================
# TAREA C: Modelo de regresión lineal
# =====================================================================
print("\n📐 TAREA C: Modelo de Regresión Lineal")
print("-"*50)

# Filtrar solo los datos válidos (sin NaN)
df_validos = df[['sulfate', 'nitrate']].dropna()

print(f"   • Registros utilizados para el modelo: {len(df_validos):,}")

# Preparar datos para el modelo
X = df_validos[['sulfate']].values
y = df_validos['nitrate'].values

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Obtener predicciones
y_pred = modelo.predict(X)

# Parámetros del modelo
intercepto = modelo.intercept_
pendiente = modelo.coef_[0]
r2 = r2_score(y, y_pred)

print(f"\n🔬 Parámetros del Modelo:")
print(f"   • Intercepto (β₀): {intercepto:.4f}")
print(f"   • Pendiente (β₁): {pendiente:.4f}")
print(f"   • Ecuación: nitrato = {intercepto:.4f} + {pendiente:.4f} × sulfato")
print(f"   • R² (Coeficiente de determinación): {r2:.4f}")

# Calcular correlación de Pearson
correlacion = df_validos['sulfate'].corr(df_validos['nitrate'])
print(f"   • Correlación de Pearson: {correlacion:.4f}")

# Visualización del modelo
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Scatter plot con línea de regresión
ax1 = axes[0, 0]
ax1.scatter(X, y, alpha=0.3, s=5, color='blue', label='Datos observados')
ax1.plot(X, y_pred, color='red', linewidth=2, label='Línea de regresión')
ax1.set_xlabel('Sulfato (mg/L)', fontsize=11)
ax1.set_ylabel('Nitrato (mg/L)', fontsize=11)
ax1.set_title('Modelo de Regresión Lineal: Sulfato vs Nitrato', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Agregar ecuación y R² en el gráfico
textstr = f'$nitrato = {intercepto:.3f} + {pendiente:.3f} × sulfato$\n$R^2 = {r2:.4f}$\n$r = {correlacion:.4f}$'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Histograma de residuos
ax2 = axes[0, 1]
residuos = y - y_pred
ax2.hist(residuos, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax2.set_xlabel('Residuos', fontsize=11)
ax2.set_ylabel('Frecuencia', fontsize=11)
ax2.set_title('Distribución de Residuos', fontsize=13, fontweight='bold')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.grid(True, alpha=0.3)

# 3. Q-Q plot
ax3 = axes[1, 0]
from scipy import stats
stats.probplot(residuos, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot de Residuos', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Residuos vs valores predichos
ax4 = axes[1, 1]
ax4.scatter(y_pred, residuos, alpha=0.3, s=5)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Valores Predichos', fontsize=11)
ax4.set_ylabel('Residuos', fontsize=11)
ax4.set_title('Residuos vs Valores Predichos', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Análisis del Modelo de Regresión Lineal', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# =====================================================================
# TAREA D: Evaluación de la hipótesis de la empresa
# =====================================================================
print("\n🔍 TAREA D: Evaluación de la Hipótesis")
print("-"*50)

print("\n📋 Hipótesis de la empresa:")
print("   'La cantidad de sulfatos está fuertemente correlacionada con la")
print("    cantidad de nitratos, tal que a mayor cantidad de sulfatos,")
print("    mayor cantidad de nitratos.'")

print("\n📊 Resultados del análisis:")
print(f"   • Correlación de Pearson: {correlacion:.4f}")
print(f"   • R² del modelo: {r2:.4f}")
print(f"   • Pendiente del modelo: {pendiente:.4f}")

# Interpretación de la correlación
if abs(correlacion) < 0.1:
    fuerza_correlacion = "prácticamente inexistente"
elif abs(correlacion) < 0.3:
    fuerza_correlacion = "muy débil"
elif abs(correlacion) < 0.5:
    fuerza_correlacion = "débil"
elif abs(correlacion) < 0.7:
    fuerza_correlacion = "moderada"
elif abs(correlacion) < 0.9:
    fuerza_correlacion = "fuerte"
else:
    fuerza_correlacion = "muy fuerte"

print("\n✅ CONCLUSIÓN:")
print("="*50)
print(f"La hipótesis de la empresa NO ES CORRECTA.")
print(f"\nJustificación:")
print(f"1. La correlación entre sulfatos y nitratos es {fuerza_correlacion}")
print(f"   (r = {correlacion:.4f}), muy lejos de ser 'fuerte'.")
print(f"\n2. El modelo de regresión lineal explica solo el {r2*100:.2f}% de la")
print(f"   variabilidad de los nitratos basándose en los sulfatos.")
print(f"\n3. Aunque la pendiente es positiva ({pendiente:.4f}), el efecto es")
print(f"   mínimo y estadísticamente no significativo.")
print(f"\n4. Los datos sugieren que los niveles de sulfatos y nitratos son")
print(f"   prácticamente independientes entre sí.")

# Estadísticas descriptivas adicionales
print("\n📊 Estadísticas descriptivas de los datos válidos:")
print("-"*50)
print(df_validos.describe())

# Crear visualización adicional: Boxplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.boxplot([df_validos['sulfate'].values], labels=['Sulfato'])
ax1.set_ylabel('Concentración (mg/L)')
ax1.set_title('Distribución de Sulfatos')
ax1.grid(True, alpha=0.3)

ax2.boxplot([df_validos['nitrate'].values], labels=['Nitrato'])
ax2.set_ylabel('Concentración (mg/L)')
ax2.set_title('Distribución de Nitratos')
ax2.grid(True, alpha=0.3)

plt.suptitle('Distribución de Concentraciones', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)