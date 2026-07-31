"""
Funciones de reportes y visualización para métricas de segmentación de nubes de puntos LiDAR.

Este módulo contiene funciones para crear tablas, gráficos y reportes completos
de las métricas de evaluación, incluyendo matrices de confusión, histogramas
de distancias y tablas resumen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from typing import Dict, List, Optional, Tuple


def create_iou_table(metrics: Dict) -> pd.DataFrame:
    """
    Genera un DataFrame con mIoU y IoU por clase.
    
    Args:
        metrics: Diccionario de métricas que contiene 'miou' e 'iou_per_class'
        
    Returns:
        DataFrame con métricas de IoU formateadas
    """
    class_names = metrics.get('class_names', {i: f"Clase {i}" for i in metrics['iou_per_class']})
    rows = [{'Métrica': 'mIoU', 'Valor': f"{metrics['miou']*100:.2f}%"}]
    for cid, iou in metrics['iou_per_class'].items():
        rows.append({
            'Métrica': f"IoU {class_names.get(cid, cid)}",
            'Valor': f"{iou*100:.2f}%"
        })
    return pd.DataFrame(rows)


def create_f1_table(metrics: Dict) -> pd.DataFrame:
    """
    Genera un DataFrame con F1-score por clase y overall accuracy.
    
    Args:
        metrics: Diccionario de métricas que contiene F1 scores
        
    Returns:
        DataFrame con métricas de F1-score formateadas
    """
    class_names = metrics.get('class_names', {i: f"Clase {i}" for i in metrics['f1_per_class']})
    rows = [
        {'Métrica': 'F1-Score', 'Valor': f"{metrics['mean_f1']*100:.2f}%"},
        {'Métrica': 'Overall Accuracy', 'Valor': f"{metrics['overall_accuracy']*100:.2f}%"}
    ]
    for cid, f1 in metrics['f1_per_class'].items():
        rows.append({
            'Métrica': f"F1 {class_names.get(cid, cid)}",
            'Valor': f"{f1*100:.2f}%"
        })
    return pd.DataFrame(rows)


def create_precision_table(metrics: Dict) -> pd.DataFrame:
    """
    Genera un DataFrame con Precision por clase.
    
    Args:
        metrics: Diccionario de métricas que contiene precision scores
        
    Returns:
        DataFrame con métricas de precision formateadas
    """
    class_names = metrics.get('class_names', {i: f"Clase {i}" for i in metrics['precision_per_class']})
    rows = [
        {'Métrica': 'Precision', 'Valor': f"{metrics['mean_precision']*100:.2f}%"},
    ]
    for cid, precision in metrics['precision_per_class'].items():
        rows.append({
            'Métrica': f"Precision {class_names.get(cid, cid)}",
            'Valor': f"{precision*100:.2f}%"
        })
    return pd.DataFrame(rows)


def create_recall_table(metrics: Dict) -> pd.DataFrame:
    """
    Genera un DataFrame con Recall por clase.
    
    Args:
        metrics: Diccionario de métricas que contiene recall scores
        
    Returns:
        DataFrame con métricas de recall formateadas
    """
    class_names = metrics.get('class_names', {i: f"Clase {i}" for i in metrics['recall_per_class']})
    rows = [
        {'Métrica': 'Recall', 'Valor': f"{metrics['mean_recall']*100:.2f}%"},
    ]
    for cid, recall in metrics['recall_per_class'].items():
        rows.append({
            'Métrica': f"Recall {class_names.get(cid, cid)}",
            'Valor': f"{recall*100:.2f}%"
        })
    return pd.DataFrame(rows)


def create_distance_table(metrics: Dict) -> pd.DataFrame:
    """Create a paper-style MDE, rho, and mu summary table."""
    class_names = metrics.get(
        'class_names',
        {i: f"Clase {i}" for i in metrics.get('mean_dist_errors', {}).keys()}
    )
    rows = [
        {
            'Class': 'Macro average',
            'MDE (m)': metrics.get('mmde', metrics.get('avg_error_distance', 0)),
            'rho (%)': np.nan,
            'mu (m)': np.nan,
            'Distant errors': metrics.get(
                'total_distant_errors',
                metrics.get('total_critic_points', 0),
            ),
        }
    ]

    mean_distances = metrics.get('mean_dist_errors', {})
    mean_non_distant = metrics.get(
        'mean_non_distant_errors',
        metrics.get('mean_non_critic_errors', {}),
    )
    distant_counts = metrics.get(
        'distant_errors_per_class',
        metrics.get('critic_points_per_class', {}),
    )
    gt_counts = metrics.get('gt_count_per_class', {})

    for cid, dist in mean_distances.items():
        name = class_names.get(cid, cid)
        count = distant_counts.get(cid, 0)
        denominator = gt_counts.get(cid, 0)
        rho = 100 * count / denominator if denominator else 0.0
        rows.append({
            'Class': name,
            'MDE (m)': dist,
            'rho (%)': rho,
            'mu (m)': mean_non_distant.get(cid, 0),
            'Distant errors': count,
        })

    table = pd.DataFrame(rows)
    for column in ('MDE (m)', 'rho (%)', 'mu (m)'):
        table[column] = table[column].round(3)
    return table


def plot_confusion_matrix(
    metrics: Dict, 
    figsize: Tuple[int, int] = (7, 7), 
    title: str = 'Matriz de Confusión', 
    show_percentages: bool = True, 
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Dibuja la matriz de confusión (conteos + %). 
    
    Args:
        metrics: Dict que debe contener 'confusion_matrix' y 'class_names'
        figsize: Tamaño de la figura
        title: Título del gráfico
        show_percentages: Si mostrar porcentajes en las celdas
        cmap: Mapa de colores para el heatmap
        
    Returns:
        Figura de matplotlib
    """
    cm = metrics['confusion_matrix']
    names = metrics['class_names']
    classes = list(names.keys())
    
    plt.figure(figsize=figsize)
    # Manejar división por cero cuando alguna clase no tiene puntos en GT
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm.astype(float)), where=row_sums!=0) * 100
    ax = sns.heatmap(cm_pct, annot=cm, fmt='d', cmap=cmap,
                     xticklabels=[names[c] for c in classes],
                     yticklabels=[names[c] for c in classes],
                     cbar=False, linewidths=0.5, linecolor='gray')
    
    if show_percentages:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i,j]:
                    color = 'white' if cm_pct[i,j]>50 else 'black'
                    ax.text(j+0.5, i+0.7, f"{cm_pct[i,j]:.1f}%", ha='center', color=color, fontsize=9)
    
    ax.set(title=title, xlabel='Predicción', ylabel='Ground Truth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.tight_layout()
    return plt.gcf()


def save_confusion_matrix_csv(metrics: Dict, filepath: str) -> None:
    """
    Exporta la matriz de confusión a CSV con filas/columnas etiquetadas.
    
    Args:
        metrics: Dict que contiene 'confusion_matrix' y 'class_names'
        filepath: Ruta donde guardar el archivo CSV
    """
    ids = sorted(metrics['class_names'])
    df = pd.DataFrame(
        metrics['confusion_matrix'], 
        index=[metrics['class_names'][i] for i in ids], 
        columns=[metrics['class_names'][i] for i in ids]
    )
    df.index.name = 'GT / Pred'
    df.to_csv(filepath)


def plot_distance_histogram(
    metrics: Dict, 
    class_id: Optional[int] = None, 
    error_type: str = 'fp', 
    figsize: Tuple[int, int] = (10, 6), 
    fontsize: int = 10,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    show_classes: bool = True,
    class_names: Optional[Dict[int, str]] = None,
    class_distance_limits: Optional[Dict[int, float]] = None
) -> plt.Figure:
    """
    Crea un histograma de distribución de distancias de error para falsos positivos o falsos negativos.
    """
    # Configuración inicial
    class_names = class_names or metrics.get('class_names', {})
    class_distance_limits = class_distance_limits or {}
    
    # Determinar fuente de datos según tipo de error
    is_fp = error_type == 'fp'
    data = metrics if is_fp else metrics.get('fn_metrics', {})
    legend_title = "Clase real" if is_fp else "Clase predicha"
    title_type = "Falsos Positivos" if is_fp else "Falsos Negativos"
    
    # Extraer distancias y clases
    dist_dict = data.get(
        'error_distances' if is_fp else 'dist_errors',
        data.get('dist_errors', {}),
    )
    assoc_dict = data.get('true_classes' if is_fp else 'pred_classes', {})
    
    # Datos específicos de clase o globales
    if class_id is not None:
        distances = np.array(dist_dict.get(class_id, []), dtype=float)
        classes = np.array(assoc_dict.get(class_id, []), dtype=int) if show_classes else np.array([])
        title_cls = class_names.get(class_id, f"Clase {class_id}")
    else:
        # Construir arrays sincronizados: cada distancia debe tener su clase correspondiente
        all_dists = []
        all_classes = []
        for cid, dist_vals in dist_dict.items():
            class_vals = assoc_dict.get(cid, [])
            # Asegurar que dist_vals y class_vals tengan la misma longitud
            min_len = min(len(dist_vals), len(class_vals)) if show_classes else len(dist_vals)
            all_dists.extend(dist_vals[:min_len])
            if show_classes:
                all_classes.extend(class_vals[:min_len])
        distances = np.array(all_dists, dtype=float)
        classes = np.array(all_classes, dtype=int) if show_classes else np.array([])
        title_cls = "todas las clases"
    
    if not distances.size:
        # Crear figura vacía si no hay datos
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No hay datos de error', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Distribución de Distancias de Error - {title_type} - {title_cls}")
        return fig
    
    # Verificar y ajustar tamaño de classes para que coincida con distances
    if show_classes and classes.size != distances.size:
        # Si hay desajuste, truncar o rellenar según corresponda
        if classes.size > distances.size:
            classes = classes[:distances.size]
        elif classes.size < distances.size:
            # Si faltan clases, crear array vacío (no se mostrarán clases en el histograma)
            classes = np.array([], dtype=int)
            show_classes = False
    
    # Generar estadísticas (siempre con datos completos)
    class_limit = class_distance_limits.get(class_id) if class_id is not None else None
    if class_limit is not None:
        critic_count = np.sum(distances > class_limit)
        critic_pct = critic_count / distances.size * 100
        stats_text = (f"Total de errores: {distances.size}\n"
                     f"Error medio (m): {distances.mean():.2f}\n"
                     f"Mediana (m): {np.median(distances):.2f}\n"
                     f"Máximo (m): {distances.max():.2f}\n"
                     f"Críticos: {critic_count} ({critic_pct:.1f}%)")
    else:
        stats_text = (f"Total de errores: {distances.size}\n"
                     f"Error medio (m): {distances.mean():.2f}\n"
                     f"Mediana (m): {np.median(distances):.2f}\n"
                     f"Máximo (m): {distances.max():.2f}\n"
                     f"Sin límite crítico definido")
    
    # Filtrar datos para visualización si es necesario
    viz_mask = np.ones(len(distances), dtype=bool)
    if min_distance is not None:
        viz_mask &= distances >= min_distance
    if max_distance is not None:
        viz_mask &= distances <= max_distance
    
    distances_viz = distances[viz_mask]
    classes_viz = classes[viz_mask] if show_classes and classes.size == distances.size else np.array([])
    
    if distances_viz.size != distances.size:
        stats_text += f"\n[Zoom: {distances_viz.size}/{distances.size} errores mostrados]"
    
    # Calcular bins
    if distances_viz.size > 0:
        bin_start = np.floor(distances_viz.min()) if min_distance is None else min_distance
        bin_end = np.ceil(distances_viz.max()) + 1
        
        # Expandir para distancias iguales
        if distances_viz.min() == distances_viz.max():
            center = distances_viz.min()
            bin_start = max(0, np.floor(center - 1))
            bin_end = np.ceil(center + 2)
        
        # Expandir para incluir límites críticos
        if class_distance_limits:
            limits = [class_distance_limits.get(class_id)] if class_id is not None else list(class_distance_limits.values())
            limits = [l for l in limits if l is not None]
            if limits:
                bin_start = min(bin_start, np.floor(min(limits)))
                bin_end = max(bin_end, np.ceil(max(limits)) + 1)
        
        bins = np.arange(bin_start, bin_end, 1.0)
        
        # Asegurar mínimo de bins para visualización
        if len(bins) < 4:  # Menos de 3 bins
            center = (bin_start + bin_end - 1) / 2
            bin_start = max(0, center - 2)
            bin_end = center + 3
            bins = np.arange(bin_start, bin_end, 1.0)
    else:
        bins = np.arange(0, 10, 1.0)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Histograma
    if show_classes and classes_viz.size:
        uniq, counts = np.unique(classes_viz, return_counts=True)
        order = np.argsort(-counts)
        stacks = [distances_viz[classes_viz == uniq[i]] for i in order]
        labels = [f"{class_names.get(uniq[i], str(uniq[i]))} ({counts[i]/len(distances_viz)*100:.1f}%)" 
                 for i in order]
        ax.hist(stacks, bins=bins, stacked=True, alpha=0.7, label=labels, edgecolor='black')
    else:
        ax.hist(distances_viz, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    
    # Marcador para distancias iguales
    if distances_viz.size > 0 and distances_viz.min() == distances_viz.max():
        unique_dist = distances_viz[0]
        ax.axvline(unique_dist, color='darkblue', linestyle='-', linewidth=3, alpha=0.8,
                  label=f'Todas = {unique_dist:.2f}m')
    
    # Líneas de límites críticos
    x_min, x_max = bins[0], bins[-1]
    if class_id is not None and class_limit is not None and x_min <= class_limit <= x_max:
        ax.axvline(class_limit, color='red', linestyle='--', linewidth=2,
                  label=f'Límite Crítico ({class_limit}m)')
    elif class_id is None and class_distance_limits:
        colors = ['red', 'orange', 'purple', 'brown', 'pink', 'green']
        unique_limits = {}
        for cid, limit in class_distance_limits.items():
            if limit is not None and x_min <= limit <= x_max:
                if limit not in unique_limits:
                    unique_limits[limit] = []
                unique_limits[limit].append(class_names.get(cid, f"Clase {cid}"))
        
        for i, (limit, names) in enumerate(unique_limits.items()):
            if i < len(colors):
                name_str = ", ".join(names[:2]) + (" ..." if len(names) > 2 else "")
                ax.axvline(limit, color=colors[i], linestyle='--', linewidth=2,
                          label=f'Límite {limit}m ({name_str})')
    
    # Leyenda arriba a la derecha
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        title = legend_title if class_id is not None else ('Clases y Límites' if len(handles) > 1 else None)
        legend = ax.legend(handles, labels, title=title, loc='upper right', fontsize=fontsize-1)
        if legend and legend.get_title():
            legend.get_title().set_fontsize(fontsize)
    
    # Títulos y etiquetas
    plot_title = f"Distribución de Distancias de Error - {title_type} - {title_cls}"
    ax.set_title(plot_title, fontsize=fontsize*1.6, pad=20, fontweight='bold')
    ax.set_xlabel('Distancia de Error (m)', fontsize=fontsize*1.2, labelpad=10)
    ax.set_ylabel('Frecuencia', fontsize=fontsize*1.2, labelpad=10)
    ax.tick_params(labelsize=fontsize)
    
    # Caja de estadísticas abajo a la derecha con fondo blanco
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=fontsize-2,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    return fig


def create_error_contribution_table(union_summary: Dict) -> pd.DataFrame:
    """
    Create table of error contribution by model.
    
    Args:
        union_summary: Resumen de contribución de errores por modelo
        
    Returns:
        DataFrame con estadísticas de contribución de errores
    """
    rows = []
    total_all_errors = sum(stats['total_errors'] for stats in union_summary['model_comparison'].values())
    
    for model, stats in union_summary['model_comparison'].items():
        error_share = stats['total_errors'] / total_all_errors * 100 if total_all_errors > 0 else 0.0
        rows.append({
            'Modelo': model,
            'Errores Totales (% sobre el total)': f"{stats['total_errors']} ({error_share:.2f}%)",
            'Errores Únicos (% sobre sus errores totales)': f"{stats['unique_errors']} ({stats['unique_error_rate']:.1f}%)",
        })
    
    return pd.DataFrame(rows)


def display_evaluation_results(
    global_metrics: Dict,
    individual_metrics: Optional[Dict] = None,
    save_path: Optional[str] = None,
    markdown: bool = False,
    dpi: int = 300,
    class_distance_limits: Optional[Dict[int, float]] = None,
    class_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Muestra las métricas principales y, si se indica, guarda resultados completos.

    Args:
        global_metrics: Métricas globales calculadas
        individual_metrics: Métricas por archivo individual (opcional)
        save_path: Ruta base donde guardar resultados (opcional)
        markdown: Si True, imprime en formato Markdown para CLI. Si False, muestra en notebook.
        dpi: Resolución para guardar figuras
        class_distance_limits: Umbrales críticos de distancia por clase
        class_names: Nombres de las clases para visualización
        
    Note:
        - Si markdown=True: imprime tablas IoU, F1 y distance en Markdown (CLI).
        - Si markdown=False: muestra DataFrames y figuras (Notebook).
        - Si save_path se proporciona: guarda todo (tablas, matrices y histogramas) bajo esa carpeta.
    """
    # 1. Tablas principales
    iou_df  = create_iou_table(global_metrics)
    f1_df   = create_f1_table(global_metrics)
    recall_df = create_recall_table(global_metrics)
    precision_df = create_precision_table(global_metrics)
    dist_df = create_distance_table(global_metrics)

    if markdown:
        print("### IoU Metrics")
        print(iou_df.to_markdown(index=False), end="\n\n")
        print("### F1 Metrics")
        print(f1_df.to_markdown(index=False), end="\n\n")
        print("### Distance Metrics")
        print(dist_df.to_markdown(index=False))
    else:
        display(iou_df.style.hide(axis='index'))
        display(f1_df.style.hide(axis='index'))
        display(dist_df.style.hide(axis='index'))

    # 2. Mostrar histogramas globales
    for err in ('fp', 'fn'):
        key = 'dist_errors' if err=='fp' else 'fn_metrics'
        if err=='fn' and key not in global_metrics:
            continue
        fig = plot_distance_histogram(global_metrics, error_type=err, 
                                     class_distance_limits=class_distance_limits,
                                     class_names=class_names)
        if not markdown:
            plt.show()
        plt.close(fig)

    # 3. Guardado
    if save_path:
        # save_path = os.path.join(save_path, 'evaluation_results')
        os.makedirs(save_path, exist_ok=True)
        
        # a) Guardar tablas globales
        iou_df.to_csv(os.path.join(save_path, 'global_iou.csv'), index=False)
        f1_df.to_csv(os.path.join(save_path, 'global_f1.csv'), index=False)
        dist_df.to_csv(os.path.join(save_path, 'global_distance.csv'), index=False)
        recall_df.to_csv(os.path.join(save_path, 'global_recall.csv'), index=False)
        precision_df.to_csv(os.path.join(save_path, 'global_precision.csv'), index=False)
        
        # b) Confusion matrix
        fig_cm = plot_confusion_matrix(global_metrics)
        fig_cm.savefig(os.path.join(save_path, 'global_confusion_matrix.png'), dpi=dpi, bbox_inches='tight')
        save_confusion_matrix_csv(global_metrics, os.path.join(save_path, 'global_confusion_matrix.csv'))
        plt.close(fig_cm)
        
        # c) Histogramas global y por clase
        for err in ('fp', 'fn'):
            if err=='fn' and 'fn_metrics' not in global_metrics:
                continue
            hist_dir = os.path.join(save_path, f'hist_{err}')
            os.makedirs(hist_dir, exist_ok=True)
            
            # global
            fig = plot_distance_histogram(global_metrics, error_type=err,
                                         class_distance_limits=class_distance_limits,
                                         class_names=class_names)
            fig.savefig(os.path.join(hist_dir, 'global.png'), dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            # por clase
            dist_dict = global_metrics['dist_errors'] if err=='fp' else global_metrics['fn_metrics']['dist_errors']
            for cid, vals in dist_dict.items():
                if not vals:
                    continue
                fig = plot_distance_histogram(global_metrics, class_id=cid, error_type=err,
                                             class_distance_limits=class_distance_limits,
                                             class_names=class_names)
                safe = str(cid)
                fig.savefig(os.path.join(hist_dir, f'class_{safe}.png'), dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
        # d) Individuales
        if individual_metrics:
            ind_dir = os.path.join(save_path, 'individual')
            for name, mets in individual_metrics.items():
                sub = os.path.join(ind_dir, os.path.splitext(name)[0])
                os.makedirs(sub, exist_ok=True)
                
                # tablas
                create_iou_table(mets).to_csv(os.path.join(sub, 'iou.csv'), index=False)
                create_f1_table(mets).to_csv(os.path.join(sub, 'f1.csv'), index=False)
                create_distance_table(mets).to_csv(os.path.join(sub, 'distance.csv'), index=False)
                create_recall_table(mets).to_csv(os.path.join(sub, 'recall.csv'), index=False)
                create_precision_table(mets).to_csv(os.path.join(sub, 'precision.csv'), index=False)
                
                # cm
                fig = plot_confusion_matrix(mets)
                fig.savefig(os.path.join(sub, 'confusion_matrix.png'), dpi=dpi, bbox_inches='tight')
                save_confusion_matrix_csv(mets, os.path.join(sub, 'confusion_matrix.csv'))
                plt.close(fig)
                
                # hist por err
                for err in ('fp','fn'):
                    key = 'dist_errors' if err=='fp' else 'fn_metrics'
                    if err=='fn' and 'fn_metrics' not in mets:
                        continue
                    hd = os.path.join(sub, f'hist_{err}')
                    os.makedirs(hd, exist_ok=True)
                    
                    # global muestra
                    fig = plot_distance_histogram(mets, error_type=err,
                                                 class_distance_limits=class_distance_limits,
                                                 class_names=class_names)
                    fig.savefig(os.path.join(hd, 'global.png'), dpi=dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    # por clase
                    dct = mets['dist_errors'] if err=='fp' else mets['fn_metrics']['dist_errors']
                    for cid, vals in dct.items():
                        if not vals:
                            continue
                        fig = plot_distance_histogram(mets, class_id=cid, error_type=err,
                                                     class_distance_limits=class_distance_limits,
                                                     class_names=class_names)
                        fig.savefig(os.path.join(hd, f'class_{cid}.png'), dpi=dpi, bbox_inches='tight')
                        plt.close(fig)
