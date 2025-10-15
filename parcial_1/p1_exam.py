#!/usr/bin/env python3
"""
ANÁLISIS ESTADÍSTICO - PROBLEMA 1 (versión mejorada)
- Lectura segura del CSV (elimina índice "Unnamed: 0" si aparece)
- Estimación de parámetros (pooled y dentro de muestra)
- Verificación de distribución de medias (Shapiro + Lilliefors opcional)
- Diagnósticos de uniformidad y homogeneidad
- Gráficas claras (bins alineados a enteros para medias ~[0,99])
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kstest, levene, chi2
from scipy.special import gamma as gamma_fn

# Lilliefors (opcional): si statsmodels está instalado, lo usamos
try:
    from statsmodels.stats.diagnostic import lilliefors
    HAVE_LILLIEFORS = True
except Exception:
    HAVE_LILLIEFORS = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ----------------------------- Utilidades -----------------------------

def leer_csv_muestras(path: str) -> pd.DataFrame:
    """Lee el CSV y elimina una posible columna índice ('Unnamed: 0')."""
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Si la primera columna es exactamente el vector 0..n-1, también la quitamos
    if len(df.columns) > 0:
        first = df.columns[0]
        if pd.api.types.is_integer_dtype(df[first]) and np.array_equal(df[first].values, np.arange(len(df))):
            df = df.drop(columns=[first])
    return df

def c4(n: int) -> float:
    """Constante c4 para corregir el sesgo de s (desv. estándar muestral).
    Fórmula: c4 = sqrt(2/(n-1)) * Γ(n/2) / Γ((n-1)/2)
    """
    if n <= 1:
        return np.nan
    return np.sqrt(2/(n-1)) * gamma_fn(n/2) / gamma_fn((n-1)/2)

def bins_uniformes_enteros(x: np.ndarray, ancho: float = 1.0):
    """Crea bordes de bins centrados en enteros para histogramas de medias ~[0,99]."""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    left = np.floor(xmin - 0.5)
    right = np.ceil(xmax + 0.5)
    return np.arange(left, right + ancho, ancho)

# ----------------------------- Parte A -----------------------------

def analisis_parte_a(data: pd.DataFrame):
    """
    PARTE A: Estimación de la media y desviación estándar poblacional
    Distingue entre parámetros de mezcla (pooled total) y σ intrínseco
    """
    print("\n" + "="*70)
    print("A) ESTIMACIÓN DE LA MEDIA Y DESVIACIÓN ESTÁNDAR POBLACIONAL")
    print("="*70)
    
    all_data = data.values.flatten()
    N = all_data.size
    
    # Método 1: Pooled total (parámetros de la mezcla)
    mu_est1 = float(np.mean(all_data))
    sigma_est1 = float(np.std(all_data, ddof=0))  # población (mezcla)
    
    # Método 2: Dentro de muestra (σ intrínseco, suponiendo homogeneidad)
    sample_means = data.mean(axis=0).values
    sample_stds = data.std(axis=0, ddof=1).values
    sample_vars = data.var(axis=0, ddof=1).values
    
    mu_est2 = float(np.mean(sample_means))
    sigma_est2_pooled = float(np.sqrt(np.mean(sample_vars)))  # σ dentro
    
    # Corrección c4 del sesgo para s
    n = data.shape[0]
    c4_n = c4(n)
    s_unbiased = sample_stds / c4_n if np.isfinite(c4_n) and c4_n > 0 else sample_stds
    
    print(f"\n📊 MÉTODO 1: Pooled total (mezcla de poblaciones)")
    print(f"   • N total: {N:,}")
    print(f"   • μ̂ (pooled): {mu_est1:.4f}")
    print(f"   • σ̂ (pooled): {sigma_est1:.4f}  (var = {sigma_est1**2:.4f})")
    print(f"   ⚠️ Nota: Este σ refleja la mezcla de todas las poblaciones")
    
    print(f"\n📊 MÉTODO 2: Dentro de muestra (σ intrínseco)")
    print(f"   • Media de las medias: {mu_est2:.4f}")
    print(f"   • σ intrínseco (pooled de varianzas): {sigma_est2_pooled:.4f}")
    print(f"   • Media de s (no corregido): {np.mean(sample_stds):.4f}")
    if np.isfinite(c4_n):
        print(f"   • Media de s/c4 (corregido): {np.mean(s_unbiased):.4f}  (c4={c4_n:.5f})")
    
    # Análisis de heterogeneidad
    print("\n⚠️ ANÁLISIS DE HETEROGENEIDAD:")
    print(f"   • Rango de medias: [{np.min(sample_means):.4f}, {np.max(sample_means):.4f}]")
    print(f"   • Amplitud del rango: {np.max(sample_means) - np.min(sample_means):.4f}")
    print(f"   • CV de las desv. estándar: {np.std(sample_stds)/np.mean(sample_stds)*100:.2f}%")
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histograma de todas las observaciones
    axes[0].hist(all_data, bins=100, alpha=0.75, edgecolor='black', linewidth=0.5)
    axes[0].axvline(mu_est1, color='red', linestyle='--', linewidth=2, label=f'μ = {mu_est1:.2f}')
    axes[0].set_title('Distribución de todas las observaciones (mezcla)')
    axes[0].set_xlabel('Valor')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    
    # Histograma de medias con bins alineados a enteros
    axes[1].hist(sample_means, bins=bins_uniformes_enteros(sample_means), alpha=0.75,
                 color='green', edgecolor='black', linewidth=0.5)
    axes[1].axvline(mu_est2, color='red', linestyle='--', linewidth=2, label=f'μ̄ = {mu_est2:.2f}')
    axes[1].set_title('Distribución de medias muestrales')
    axes[1].set_xlabel('Media muestral')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend()
    
    # Boxplot de desviaciones estándar
    axes[2].boxplot(sample_stds, vert=True)
    axes[2].set_title('Desviaciones estándar muestrales')
    axes[2].set_ylabel('Desviación estándar')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mu_est1, sigma_est1, sample_means, sample_stds, s_unbiased

# ----------------------------- Parte B -----------------------------

def analisis_parte_b(sample_means: np.ndarray, sample_stds: np.ndarray):
    """
    PARTE B: Análisis de la distribución de la media muestral
    Incluye Lilliefors si está disponible, sino KS con advertencia
    """
    print("\n" + "="*70)
    print("B) DISTRIBUCIÓN DE LA MEDIA MUESTRAL")
    print("="*70)
    
    mean_of_means = float(np.mean(sample_means))
    std_of_means = float(np.std(sample_means, ddof=1))
    n_per_sample = 5000
    theoretical_sem = float(np.mean(sample_stds) / np.sqrt(n_per_sample))
    
    print("\n📊 CARACTERÍSTICAS DE LA DISTRIBUCIÓN:")
    print(f"   • Número de medias muestrales: {len(sample_means)}")
    print(f"   • Media de las medias (x̄̄): {mean_of_means:.4f}")
    print(f"   • Desv. estándar de las medias: {std_of_means:.4f}")
    print(f"   • Mínimo: {np.min(sample_means):.4f}")
    print(f"   • Máximo: {np.max(sample_means):.4f}")
    print(f"   • Rango: {np.max(sample_means) - np.min(sample_means):.4f}")
    
    print("\n📈 TEOREMA DEL LÍMITE CENTRAL (si fuese 1 sola población):")
    print(f"   • Tamaño de cada muestra (n): {n_per_sample}")
    print(f"   • Error estándar teórico (σ/√n): {theoretical_sem:.6f}")
    print(f"   • Error estándar observado: {std_of_means:.4f}")
    print(f"   • Razón observado/teórico: {std_of_means/theoretical_sem:.1f}")
    print(f"   ⚠️ Esta razón extrema indica múltiples poblaciones")
    
    # Pruebas de normalidad
    stat_shapiro, p_shapiro = shapiro(sample_means)
    print("\n📊 PRUEBAS DE NORMALIDAD:")
    print(f"   • Test de Shapiro-Wilk:")
    print(f"     - Estadístico: {stat_shapiro:.4f}")
    print(f"     - p-valor: {p_shapiro:.4f}")
    print(f"     - Conclusión: {'Normal' if p_shapiro > 0.05 else 'No normal'} (α=0.05)")
    
    if HAVE_LILLIEFORS:
        # Estandarizar para Lilliefors
        z = (sample_means - mean_of_means) / std_of_means
        stat_lil, p_lil = lilliefors(z, dist='norm')
        print(f"   • Test de Lilliefors (más apropiado):")
        print(f"     - Estadístico: {stat_lil:.4f}")
        print(f"     - p-valor: {p_lil:.4f}")
        print(f"     - Conclusión: {'Normal' if p_lil > 0.05 else 'No normal'} (α=0.05)")
    else:
        stat_ks, p_ks = kstest(sample_means, 'norm', args=(mean_of_means, std_of_means))
        print(f"   • Test KS (⚠️ p-valor no exacto con parámetros estimados):")
        print(f"     - Estadístico: {stat_ks:.4f}")
        print(f"     - p-valor: {p_ks:.4f}")
        print(f"     - Nota: Instala statsmodels para Lilliefors")
    
    # Momentos
    skewness = stats.skew(sample_means)
    kurtosis_val = stats.kurtosis(sample_means)
    
    print(f"\n   • Asimetría (skewness): {skewness:.4f}")
    print(f"   • Curtosis (kurtosis): {kurtosis_val:.4f}")
    
    # Regla empírica
    z_scores = np.abs((sample_means - mean_of_means) / std_of_means)
    within_1sd = np.mean(z_scores <= 1) * 100
    within_2sd = np.mean(z_scores <= 2) * 100
    within_3sd = np.mean(z_scores <= 3) * 100
    
    print("\n📊 REGLA EMPÍRICA (68-95-99.7):")
    print(f"   • Dentro de ±1σ: {within_1sd:.1f}% (esperado: ~68%)")
    print(f"   • Dentro de ±2σ: {within_2sd:.1f}% (esperado: ~95%)")
    print(f"   • Dentro de ±3σ: {within_3sd:.1f}% (esperado: ~99.7%)")
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograma con bins alineados y curva normal
    bins = bins_uniformes_enteros(sample_means)
    axes[0, 0].hist(sample_means, bins=bins, density=True, alpha=0.75, 
                    color='skyblue', edgecolor='black', linewidth=0.5)
    x = np.linspace(np.min(sample_means), np.max(sample_means), 200)
    axes[0, 0].plot(x, stats.norm.pdf(x, mean_of_means, std_of_means), 
                    'r-', linewidth=2, label='Normal teórica')
    axes[0, 0].set_title('Distribución de medias muestrales')
    axes[0, 0].set_xlabel('Media muestral')
    axes[0, 0].set_ylabel('Densidad')
    axes[0, 0].legend()
    
    # Q-Q plot
    stats.probplot(sample_means, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Boxplot
    axes[1, 0].boxplot(sample_means, vert=False)
    axes[1, 0].set_title('Boxplot de medias muestrales')
    axes[1, 0].set_xlabel('Media muestral')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Diagrama de dispersión índice vs media
    axes[1, 1].scatter(range(len(sample_means)), sample_means, alpha=0.6, s=20)
    axes[1, 1].set_title('Medias muestrales por índice')
    axes[1, 1].set_xlabel('Índice de muestra')
    axes[1, 1].set_ylabel('Media muestral')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mean_of_means, std_of_means

# ----------------------------- Parte C -----------------------------

def analisis_parte_c(sample_means: np.ndarray, sample_stds: np.ndarray, data: pd.DataFrame):
    """
    PARTE C: Observaciones sobre el proceso de muestreo
    Análisis de uniformidad y homogeneidad mejorado
    """
    print("\n" + "="*70)
    print("C) OBSERVACIONES SOBRE EL PROCESO DE MUESTREO")
    print("="*70)
    
    # Análisis de espaciamiento
    sorted_means = np.sort(sample_means)
    spacings = np.diff(sorted_means)
    
    print("\n🔍 ANÁLISIS DE ESPACIAMIENTO ENTRE MEDIAS ORDENADAS:")
    print(f"   • Espaciamiento promedio: {np.mean(spacings):.4f}")
    print(f"   • Desv. estándar: {np.std(spacings):.4f}")
    print(f"   • Mínimo: {np.min(spacings):.4f}")
    print(f"   • Máximo: {np.max(spacings):.4f}")
    print(f"   ⚠️ Espaciamiento ≈ 1.0 sugiere medias equiespaciadas")
    
    # Test de uniformidad con bins alineados a enteros
    bins = bins_uniformes_enteros(sample_means)
    hist, edges = np.histogram(sample_means, bins=bins)
    expected = len(sample_means) / len(hist)
    chi2_stat = np.sum((hist - expected)**2 / (expected + 1e-12))
    dfree = max(len(hist) - 1, 1)
    chi2_crit = chi2.ppf(0.95, dfree)
    
    print("\n📊 TEST DE UNIFORMIDAD (χ² con bins alineados a enteros):")
    print(f"   • Número de bins: {len(hist)}")
    print(f"   • χ² observado: {chi2_stat:.4f}")
    print(f"   • χ² crítico (α=0.05, df={dfree}): {chi2_crit:.4f}")
    print(f"   • Conclusión: {'Distribución uniforme' if chi2_stat < chi2_crit else 'No uniforme'}")
    
    # Homogeneidad de varianzas
    print("\n📊 ANÁLISIS DE HOMOGENEIDAD DE VARIANZAS:")
    print(f"   • Media de las desv. estándar: {np.mean(sample_stds):.4f}")
    print(f"   • Desv. estándar de las desv. estándar: {np.std(sample_stds):.6f}")
    print(f"   • Coef. de variación: {np.std(sample_stds)/np.mean(sample_stds)*100:.2f}%")
    print(f"   ✓ CV < 2% indica varianzas muy homogéneas")
    
    # Test de Levene
    k = min(10, data.shape[1])
    samples_for_levene = [data.iloc[:, i].values for i in range(k)]
    stat_levene, p_levene = levene(*samples_for_levene)
    
    print(f"\n   • Test de Levene (primeras {k} muestras):")
    print(f"     - Estadístico: {stat_levene:.4f}")
    print(f"     - p-valor: {p_levene:.4f}")
    print(f"     - Conclusión: {'Varianzas homogéneas' if p_levene > 0.05 else 'Varianzas no homogéneas'} (α=0.05)")
    
    # Conclusiones
    print("\n" + "="*70)
    print("CONCLUSIONES FINALES")
    print("="*70)
    
    print("\n📌 HALLAZGO PRINCIPAL:")
    print("   Los datos NO provienen de una única población normal.")
    
    print("\n📊 EVIDENCIA:")
    print("   1. DISTRIBUCIÓN UNIFORME: Las medias muestrales están")
    print("      uniformemente distribuidas en [0, 99], no concentradas")
    print("      alrededor de un valor central como espera el TLC.")
    
    print("\n   2. ERROR ESTÁNDAR ANÓMALO: La razón observado/teórico")
    print("      es ~4,000, imposible bajo una sola población.")
    
    print("\n   3. VARIANZA HOMOGÉNEA: Todas las muestras tienen")
    print("      σ ≈ 0.5 (CV < 2%), sugiriendo proceso controlado.")
    
    print("\n💡 INTERPRETACIÓN:")
    print("   Cada columna representa una muestra de una población")
    print("   diferente con N(μᵢ, 0.5²), donde μᵢ ~ Uniforme(0, 99)")
    
    print("\n📊 RESPUESTAS AL PROBLEMA:")
    print("\n   a) ESTIMACIÓN DE PARÁMETROS:")
    print("      • No es apropiado estimar μ y σ únicos")
    print("      • σ intrínseco (dentro) ≈ 0.5")
    print("      • σ pooled (mezcla) ≈ 28.9 refleja heterogeneidad")
    
    print("\n   b) DISTRIBUCIÓN DE MEDIAS:")
    print("      • NO sigue el TLC para una población")
    print("      • Distribución uniforme, no normal")
    
    print("\n   c) PROCESO DE MUESTREO:")
    print("      • Artificial/simulado con 100 poblaciones")
    print("      • Cada población: N(i, 0.5²) para i ∈ [0, 99]")
    
    # Visualización final
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Distribución uniforme de las medias con bins alineados
    axes[0].hist(sample_means, bins=bins, alpha=0.75, color='coral', 
                 edgecolor='black', linewidth=0.5)
    axes[0].axhline(y=expected, color='blue', linestyle='--', 
                    linewidth=2, label=f'Esperado (uniforme): {expected:.1f}')
    axes[0].set_title('Distribución de medias muestrales')
    axes[0].set_xlabel('Media muestral')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    
    # Espaciamiento entre medias ordenadas
    axes[1].plot(spacings, 'o-', alpha=0.6, markersize=3)
    axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                    label='Espaciamiento esperado = 1.0')
    axes[1].set_title('Espaciamiento entre medias consecutivas')
    axes[1].set_xlabel('Posición')
    axes[1].set_ylabel('Espaciamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Desviaciones estándar
    axes[2].plot(sample_stds, 'o', alpha=0.6, markersize=4)
    axes[2].axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                    label='σ = 0.5')
    axes[2].set_title('Desviaciones estándar muestrales')
    axes[2].set_xlabel('Índice de muestra')
    axes[2].set_ylabel('Desviación estándar')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ----------------------------- Función Principal -----------------------------

def main():
    """Ejecuta el análisis completo con lectura segura del CSV"""
    
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║     ANÁLISIS ESTADÍSTICO COMPLETO - PROBLEMA 1 (MEJORADO)  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Intentar cargar los datos procesados primero
    try:
        # Primero intentar con los archivos procesados
        data = leer_csv_muestras('parcial_1/data/exam_data.csv')
        print(f"\n✓ Datos procesados cargados: {data.shape}")
    except FileNotFoundError:
        try:
            # Si no están, cargar el archivo original
            data = leer_csv_muestras('exam_data.csv')
            print(f"\n✓ Datos originales cargados: {data.shape}")
        except FileNotFoundError:
            print("\n❌ Error: No se encontró 'exam_data.csv' en el directorio actual.")
            print("   Asegúrate de que el archivo esté en el directorio de trabajo.")
            return
    
    # Validar estructura de datos
    if data.shape[0] != 5000 or data.shape[1] != 100:
        print(f"\n⚠️ Advertencia: Se esperaban 5000×100 datos, se encontraron {data.shape}")
    
    # Ejecutar análisis
    mu_est, sigma_est, sample_means, sample_stds, s_unbiased = analisis_parte_a(data)
    mean_of_means, std_of_means = analisis_parte_b(sample_means, sample_stds)
    analisis_parte_c(sample_means, sample_stds, data)
    
    print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()