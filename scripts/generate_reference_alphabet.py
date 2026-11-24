from __future__ import annotations

import ast
import json
import re
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


# ============================================================
# CONFIGURACIÓN
# ============================================================

# ------------------------------------------------------------
# CSV finales de landmarks
# ------------------------------------------------------------

ASL_CSV = Path(
    r"D:\Sign2SignData\processed\landmarks\asl_landmarks_final.csv"
)

LSM_CSV = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_canonical_v2_splits.csv"
)


# ------------------------------------------------------------
# Scalers utilizados por los modelos finales
# ------------------------------------------------------------

ASL_SCALER = Path(
    r"C:\Users\yaelb\Sign2Sign\models\asl\scaler_img_norm.joblib"
)

LSM_SCALER = Path(
    r"C:\Users\yaelb\Sign2Sign\models\lsm\scaler_world_norm.joblib"
)


# ------------------------------------------------------------
# Imágenes originales
# ------------------------------------------------------------

# Dataset original ASL.
ASL_IMAGE_SOURCES = [
    Path(
        r"D:\Sign2SignData\raw\asl\ASL_original.zip"
    ),
]

# Dataset original LSM.
#
# El ZIP contiene internamente:
#
# TT/
# └── dataset/
#     ├── A/
#     ├── B/
#     ├── ...
#
# Se agrega ASL_original.zip como segunda fuente porque
# E, L y Z del clasificador LSM fueron incorporadas
# provisionalmente desde ASL.
LSM_IMAGE_SOURCES = [
    Path(
        r"D:\Sign2SignData\raw\lsm\LSM_original.zip"
    ),
    Path(
        r"D:\Sign2SignData\raw\asl\ASL_original.zip"
    ),
]


# ------------------------------------------------------------
# Salida para React/Vite
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    r"C:\Users\yaelb\Sign2Sign\frontend\public\reference-alphabet"
)


# ------------------------------------------------------------
# Representación utilizada por cada modelo final
# ------------------------------------------------------------

ASL_REPRESENTATION = "img_norm"
LSM_REPRESENTATION = "world_norm"


# ------------------------------------------------------------
# Sólo seleccionar muestras de entrenamiento
# ------------------------------------------------------------

USE_ONLY_TRAIN = True


# ============================================================
# NOMBRES DE COLUMNAS POSIBLES
# ============================================================

LABEL_CANDIDATES = [
    "class",
    "label",
    "letter",
    "target",
    "class_label",
]

SPLIT_CANDIDATES = [
    "split_final",
    "split",
    "dataset_split",
    "partition",
    "subset",
]

PATH_CANDIDATES = [
    "path",
    "image_path",
    "file_path",
    "source_path",
    "original_path",
    "image",
    "filename",
    "file",
]


# ============================================================
# UTILIDADES
# ============================================================

def natural_key(text: str):
    """
    Permite ordenar columnas como:

    img_norm_0
    img_norm_1
    ...
    img_norm_10

    evitando que 10 aparezca antes que 2.
    """

    return [
        int(value)
        if value.isdigit()
        else value.lower()
        for value in re.split(
            r"(\d+)",
            text,
        )
    ]


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
):
    """
    Busca una columna ignorando mayúsculas/minúsculas.
    """

    lower_map = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    if required:

        raise ValueError(
            "\nNo se encontró ninguna de estas columnas:\n"
            f"{candidates}\n\n"
            "Columnas disponibles:\n"
            f"{list(df.columns)}"
        )

    return None


# ============================================================
# EXTRACCIÓN DE LAS 63 CARACTERÍSTICAS
# ============================================================

def get_feature_matrix(
    df: pd.DataFrame,
    representation: str,
):
    """
    Soporta dos formatos.

    FORMATO 1
    ---------
    63 columnas independientes:

    img_norm_0
    img_norm_1
    ...
    img_norm_62

    o:

    world_norm_0
    ...
    world_norm_62


    FORMATO 2
    ---------
    Una columna llamada:

    img_norm

    o:

    world_norm

    cuyo contenido sea una lista serializada de 63 valores.
    """

    # --------------------------------------------------------
    # Caso 1:
    # 63 columnas individuales
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.lower().startswith(
            representation.lower() + "_"
        )
    ]

    if feature_columns:

        feature_columns = sorted(
            feature_columns,
            key=natural_key,
        )

        X = (
            df[feature_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(
                dtype=np.float64,
            )
        )

        if X.shape[1] != 63:

            raise ValueError(
                f"\nSe encontraron {X.shape[1]} columnas "
                f"para {representation}, "
                "pero se esperaban 63.\n\n"
                "Columnas encontradas:\n"
                f"{feature_columns}"
            )

        return X, feature_columns

    # --------------------------------------------------------
    # Caso 2:
    # Una sola columna con lista serializada
    # --------------------------------------------------------

    exact_column = None

    for column in df.columns:

        if (
            column.lower()
            == representation.lower()
        ):
            exact_column = column
            break

    if exact_column is not None:

        def parse_vector(value):

            if isinstance(
                value,
                (
                    list,
                    tuple,
                    np.ndarray,
                ),
            ):

                return np.asarray(
                    value,
                    dtype=np.float64,
                )

            text = str(
                value
            ).strip()

            try:
                parsed = json.loads(
                    text
                )

            except json.JSONDecodeError:

                parsed = ast.literal_eval(
                    text
                )

            return np.asarray(
                parsed,
                dtype=np.float64,
            )

        vectors = (
            df[exact_column]
            .apply(
                parse_vector
            )
            .tolist()
        )

        X = np.vstack(
            vectors
        )

        if X.shape[1] != 63:

            raise ValueError(
                f"\n{representation} contiene "
                f"{X.shape[1]} valores, "
                "pero se esperaban 63."
            )

        return X, [exact_column]

    raise ValueError(
        f"\nNo encontré las características "
        f"'{representation}'.\n\n"
        "Columnas disponibles:\n"
        f"{list(df.columns)}"
    )


# ============================================================
# NORMALIZACIÓN DE NOMBRE DE ARCHIVO
# ============================================================

def normalize_filename(
    value: str,
) -> str:

    return (
        Path(
            str(value)
            .replace(
                "\\",
                "/",
            )
        )
        .name
        .lower()
    )


# ============================================================
# LOCALIZADOR DE IMÁGENES
# ============================================================

class ImageLocator:
    """
    Permite buscar imágenes tanto en:

    - carpetas normales;
    - archivos ZIP.

    No es necesario extraer los ZIP.
    """

    def __init__(
        self,
        sources: list[Path],
    ):

        self.files_by_name = {}

        self.zip_files = []

        for source in sources:

            # ------------------------------------------------
            # Fuente inexistente
            # ------------------------------------------------

            if not source.exists():

                print(
                    f"[WARN] Fuente no encontrada: "
                    f"{source}"
                )

                continue

            # ------------------------------------------------
            # Fuente ZIP
            # ------------------------------------------------

            if (
                source.is_file()
                and
                source.suffix.lower()
                == ".zip"
            ):

                print(
                    f"[INFO] Indexando ZIP: "
                    f"{source}"
                )

                archive = zipfile.ZipFile(
                    source,
                    "r",
                )

                index = {}

                for member in archive.namelist():

                    if member.endswith("/"):
                        continue

                    extension = (
                        Path(member)
                        .suffix
                        .lower()
                    )

                    if extension not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                    }:
                        continue

                    filename = (
                        Path(member)
                        .name
                        .lower()
                    )

                    index.setdefault(
                        filename,
                        [],
                    ).append(
                        member
                    )

                print(
                    f"[INFO] "
                    f"{len(index):,} nombres de imagen "
                    f"indexados."
                )

                self.zip_files.append(
                    (
                        source,
                        archive,
                        index,
                    )
                )

            # ------------------------------------------------
            # Fuente carpeta
            # ------------------------------------------------

            elif source.is_dir():

                print(
                    f"[INFO] Indexando carpeta: "
                    f"{source}"
                )

                for file in source.rglob("*"):

                    if not file.is_file():
                        continue

                    if file.suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                    }:
                        continue

                    self.files_by_name.setdefault(
                        file.name.lower(),
                        [],
                    ).append(
                        file
                    )

    # ========================================================
    # COPIAR IMAGEN
    # ========================================================

    def copy_image(
        self,
        original_reference: str,
        destination_without_extension: Path,
    ):

        filename = normalize_filename(
            original_reference
        )

        # ----------------------------------------------------
        # Buscar primero en carpetas normales
        # ----------------------------------------------------

        matches = (
            self.files_by_name.get(
                filename,
                [],
            )
        )

        if matches:

            source = matches[0]

            destination = (
                destination_without_extension
                .with_suffix(
                    source.suffix.lower()
                )
            )

            shutil.copy2(
                source,
                destination,
            )

            return (
                str(source),
                destination,
            )

        # ----------------------------------------------------
        # Buscar después dentro de ZIPs
        # ----------------------------------------------------

        for (
            zip_path,
            archive,
            index,
        ) in self.zip_files:

            matches = index.get(
                filename,
                [],
            )

            if not matches:
                continue

            member = matches[0]

            extension = (
                Path(member)
                .suffix
                .lower()
            )

            destination = (
                destination_without_extension
                .with_suffix(
                    extension
                )
            )

            with (
                archive.open(
                    member
                ) as source_file,
                destination.open(
                    "wb"
                ) as output_file,
            ):

                shutil.copyfileobj(
                    source_file,
                    output_file,
                )

            return (
                f"{zip_path}!{member}",
                destination,
            )

        raise FileNotFoundError(
            "\nNo se encontró la imagen original:\n"
            f"{original_reference}\n"
            f"Nombre buscado: {filename}"
        )

    # ========================================================
    # CERRAR ZIP
    # ========================================================

    def close(self):

        for (
            _,
            archive,
            _,
        ) in self.zip_files:

            archive.close()


# ============================================================
# FILTRAR TRAIN
# ============================================================

def filter_train(
    df: pd.DataFrame,
    split_column: str | None,
) -> pd.DataFrame:

    if not USE_ONLY_TRAIN:
        return df

    if split_column is None:

        print()
        print(
            "[WARN] No existe una columna de split."
        )

        print(
            "[WARN] Se utilizarán todas las muestras."
        )

        return df

    split_values = (
        df[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_mask = split_values.isin(
        {
            "train",
            "training",
        }
    )

    train_count = int(
        train_mask.sum()
    )

    if train_count == 0:

        print()
        print(
            "[WARN] Se encontró columna de split "
            "pero ninguna fila con valor TRAIN."
        )

        print(
            "[WARN] Valores encontrados:"
        )

        print(
            sorted(
                split_values
                .dropna()
                .unique()
                .tolist()
            )
        )

        print(
            "[WARN] Se utilizarán todas las muestras."
        )

        return df

    return (
        df.loc[
            train_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SELECCIÓN DE MUESTRAS REPRESENTATIVAS
# ============================================================

def select_representatives(
    *,
    language: str,
    csv_path: Path,
    scaler_path: Path,
    representation: str,
    image_sources: list[Path],
    output_dir: Path,
):

    print()
    print(
        "=" * 72
    )

    print(
        f"PROCESANDO {language}"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Verificar archivos principales
    # --------------------------------------------------------

    if not csv_path.is_file():

        raise FileNotFoundError(
            f"\nNo existe el CSV:\n"
            f"{csv_path}"
        )

    if not scaler_path.is_file():

        raise FileNotFoundError(
            f"\nNo existe el scaler:\n"
            f"{scaler_path}"
        )

    # --------------------------------------------------------
    # Leer CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    print(
        f"Filas originales: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Detectar columnas
    # --------------------------------------------------------

    label_column = find_column(
        df,
        LABEL_CANDIDATES,
    )

    split_column = find_column(
        df,
        SPLIT_CANDIDATES,
        required=False,
    )

    path_column = find_column(
        df,
        PATH_CANDIDATES,
    )

    print(
        f"Columna clase : "
        f"{label_column}"
    )

    print(
        f"Columna ruta  : "
        f"{path_column}"
    )

    print(
        f"Columna split : "
        f"{split_column}"
    )

    # --------------------------------------------------------
    # Normalizar etiquetas
    # --------------------------------------------------------

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Filtrar TRAIN
    # --------------------------------------------------------

    original_count = len(
        df
    )

    df = filter_train(
        df,
        split_column,
    )

    if len(df) != original_count:

        print(
            f"Filas TRAIN: "
            f"{len(df):,}"
        )

    # --------------------------------------------------------
    # Obtener características
    # --------------------------------------------------------

    X, feature_columns = (
        get_feature_matrix(
            df,
            representation,
        )
    )

    print(
        f"Representación: "
        f"{representation}"
    )

    print(
        f"Dimensiones: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Validación numérica
    # --------------------------------------------------------

    valid_mask = (
        np.isfinite(
            X
        )
        .all(
            axis=1
        )
    )

    if not valid_mask.all():

        removed = int(
            (
                ~valid_mask
            ).sum()
        )

        print(
            f"[WARN] Se eliminan "
            f"{removed} filas "
            "con NaN o Inf."
        )

        df = (
            df.loc[
                valid_mask
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        X = X[
            valid_mask
        ]

    # --------------------------------------------------------
    # Cargar scaler del modelo final
    # --------------------------------------------------------

    scaler = joblib.load(
        scaler_path
    )

    # --------------------------------------------------------
    # Aplicar exactamente el scaler utilizado por el modelo
    # --------------------------------------------------------

    X_scaled = (
        scaler.transform(
            X
        )
    )

    # --------------------------------------------------------
    # Preparar carpeta de salida
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Preparar localizador de imágenes
    # --------------------------------------------------------

    locator = ImageLocator(
        image_sources
    )

    manifest = []

    classes = sorted(
        df[label_column]
        .unique()
        .tolist()
    )

    print()
    print(
        "Clases encontradas: "
        + ", ".join(
            classes
        )
    )

    print()

    # ========================================================
    # CENTROIDE POR CLASE
    # ========================================================

    for class_name in classes:

        indices = np.where(
            df[label_column]
            .to_numpy()
            == class_name
        )[0]

        class_features = (
            X_scaled[
                indices
            ]
        )

        # ----------------------------------------------------
        # Centroide
        # ----------------------------------------------------

        centroid = (
            class_features.mean(
                axis=0
            )
        )

        # ----------------------------------------------------
        # Distancia de cada muestra al centroide
        # ----------------------------------------------------

        distances = np.linalg.norm(
            class_features
            - centroid,
            axis=1,
        )

        # ----------------------------------------------------
        # Muestra más cercana
        # ----------------------------------------------------

        nearest_local_index = int(
            np.argmin(
                distances
            )
        )

        nearest_global_index = int(
            indices[
                nearest_local_index
            ]
        )

        row = df.iloc[
            nearest_global_index
        ]

        original_reference = str(
            row[
                path_column
            ]
        )

        distance = float(
            distances[
                nearest_local_index
            ]
        )

        # ----------------------------------------------------
        # Copiar imagen seleccionada
        # ----------------------------------------------------

        try:

            (
                source_found,
                destination,
            ) = locator.copy_image(
                original_reference,
                output_dir
                / class_name,
            )

            print(
                f"[OK] "
                f"{class_name}: "
                f"{Path(destination).name} "
                f"(dist={distance:.4f})"
            )

            manifest.append(
                {
                    "language":
                        language,
                    "letter":
                        class_name,
                    "representation":
                        representation,
                    "distance_to_centroid":
                        distance,
                    "dataset_reference":
                        original_reference,
                    "source_found":
                        str(
                            source_found
                        ),
                    "output_file":
                        str(
                            destination
                        ),
                    "number_of_train_samples":
                        len(
                            indices
                        ),
                    "status":
                        "OK",
                }
            )

        except FileNotFoundError as error:

            print(
                f"[ERROR] "
                f"{class_name}"
            )

            print(
                error
            )

            manifest.append(
                {
                    "language":
                        language,
                    "letter":
                        class_name,
                    "representation":
                        representation,
                    "distance_to_centroid":
                        distance,
                    "dataset_reference":
                        original_reference,
                    "source_found":
                        "",
                    "output_file":
                        "",
                    "number_of_train_samples":
                        len(
                            indices
                        ),
                    "status":
                        "IMAGE_NOT_FOUND",
                }
            )

    # --------------------------------------------------------
    # Cerrar ZIPs
    # --------------------------------------------------------

    locator.close()

    # ========================================================
    # GUARDAR MANIFIESTO
    # ========================================================

    manifest_path = (
        output_dir
        / "reference_manifest.csv"
    )

    pd.DataFrame(
        manifest
    ).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Manifiesto guardado:"
    )

    print(
        manifest_path
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    success_count = sum(
        item["status"] == "OK"
        for item in manifest
    )

    failed_count = (
        len(manifest)
        - success_count
    )

    print()
    print(
        f"Referencias correctas: "
        f"{success_count}/{len(manifest)}"
    )

    if failed_count:

        print(
            f"Referencias no encontradas: "
            f"{failed_count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "GENERADOR DE ALFABETO DE REFERENCIA - SIGN2SIGN"
    )

    print(
        "=" * 72
    )

    print()
    print(
        f"scikit-learn actual: "
        f"{sklearn.__version__}"
    )

    if sklearn.__version__ != "1.6.1":

        print()
        print(
            "[WARN] Los scalers fueron generados "
            "con scikit-learn 1.6.1."
        )

        print(
            "[WARN] Se recomienda ejecutar:"
        )

        print()
        print(
            "pip install scikit-learn==1.6.1"
        )

        print()

    # --------------------------------------------------------
    # Crear raíz de salida
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # ASL
    # --------------------------------------------------------

    select_representatives(
        language="ASL",
        csv_path=ASL_CSV,
        scaler_path=ASL_SCALER,
        representation=ASL_REPRESENTATION,
        image_sources=ASL_IMAGE_SOURCES,
        output_dir=(
            OUTPUT_ROOT
            / "asl"
        ),
    )

    # --------------------------------------------------------
    # LSM
    # --------------------------------------------------------

    select_representatives(
        language="LSM",
        csv_path=LSM_CSV,
        scaler_path=LSM_SCALER,
        representation=LSM_REPRESENTATION,
        image_sources=LSM_IMAGE_SOURCES,
        output_dir=(
            OUTPUT_ROOT
            / "lsm"
        ),
    )

    # --------------------------------------------------------
    # Fin
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "PROCESO FINALIZADO"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Referencias generadas en:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()