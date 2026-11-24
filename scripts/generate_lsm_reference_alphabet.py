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


# ============================================================
# CONFIGURACIÓN
# ============================================================

LSM_CSV = Path(
    r"D:\Sign2SignData\processed\landmarks\lsm_landmarks_canonical_v2_splits.csv"
)

LSM_SCALER = Path(
    r"C:\Users\yaelb\Sign2Sign\models\lsm\scaler_world_norm.joblib"
)

LSM_ZIP = Path(
    r"D:\Sign2SignData\raw\lsm\LSM_original.zip"
)

ASL_ZIP = Path(
    r"D:\Sign2SignData\raw\asl\ASL_original.zip"
)

OUTPUT_DIR = Path(
    r"C:\Users\yaelb\Sign2Sign\frontend\public\reference-alphabet\lsm"
)

REPRESENTATION = "world_norm"

# E/L/Z provienen provisionalmente del dataset ASL.
PROVISIONAL_CLASSES = {
    "E",
    "L",
    "Z",
}


# ============================================================
# COLUMNAS POSIBLES
# ============================================================

LABEL_CANDIDATES = [
    "class",
    "label",
    "letter",
    "target",
]

PATH_CANDIDATES = [
    "path",
    "image_path",
    "file_path",
    "source_path",
]

SPLIT_CANDIDATES = [
    "split_final",
    "split",
]


# ============================================================
# UTILIDADES
# ============================================================

def natural_key(text: str):
    return [
        int(value)
        if value.isdigit()
        else value.lower()
        for value in re.split(
            r"(\d+)",
            text,
        )
    ]


def normalize_path(value: str) -> str:
    value = (
        str(value)
        .replace("\\", "/")
        .strip()
    )

    while value.startswith("./"):
        value = value[2:]

    return (
        value
        .lower()
        .strip("/")
    )


def get_filename(value: str) -> str:
    return (
        Path(
            normalize_path(value)
        )
        .name
        .lower()
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
):
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
# LEER WORLD_NORM
# ============================================================

def get_world_norm_matrix(
    df: pd.DataFrame,
):
    # --------------------------------------------------------
    # Caso 1:
    # world_norm_0 ... world_norm_62
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column.lower().startswith(
            REPRESENTATION + "_"
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
                f"Se encontraron {X.shape[1]} "
                "características world_norm, "
                "pero se esperaban 63."
            )

        return X

    # --------------------------------------------------------
    # Caso 2:
    # una columna world_norm con vector serializado
    # --------------------------------------------------------

    exact_column = None

    for column in df.columns:

        if column.lower() == REPRESENTATION:

            exact_column = column
            break

    if exact_column is None:

        raise ValueError(
            "No se encontró world_norm en el CSV."
        )

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
            f"world_norm contiene "
            f"{X.shape[1]} valores, "
            "pero se esperaban 63."
        )

    return X


# ============================================================
# ÍNDICE DE ZIP
# ============================================================

class ZipImageIndex:

    def __init__(
        self,
        zip_path: Path,
    ):
        if not zip_path.is_file():

            raise FileNotFoundError(
                f"No existe el ZIP:\n"
                f"{zip_path}"
            )

        self.zip_path = zip_path

        self.archive = zipfile.ZipFile(
            zip_path,
            "r",
        )

        self.members = []

        for member in self.archive.namelist():

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

            self.members.append(
                {
                    "original":
                        member,

                    "normalized":
                        normalize_path(
                            member
                        ),

                    "filename":
                        get_filename(
                            member
                        ),
                }
            )

        print(
            f"[INFO] {zip_path.name}: "
            f"{len(self.members):,} imágenes indexadas."
        )

    def find_by_reference(
        self,
        reference: str,
    ):
        normalized_reference = normalize_path(
            reference
        )

        filename = get_filename(
            reference
        )

        # ====================================================
        # 1. Coincidencia por ruta completa
        # ====================================================

        exact_matches = [
            item
            for item in self.members
            if (
                item["normalized"]
                == normalized_reference
                or
                item["normalized"].endswith(
                    "/" + normalized_reference
                )
            )
        ]

        if len(exact_matches) == 1:

            return exact_matches[0][
                "original"
            ]

        if len(exact_matches) > 1:

            raise RuntimeError(
                "\nMás de una coincidencia exacta "
                "para la ruta:\n"
                f"{reference}"
            )

        # ====================================================
        # 2. Fallback por nombre
        # ====================================================
        #
        # Sólo si el nombre es único en todo el ZIP.
        # ====================================================

        filename_matches = [
            item
            for item in self.members
            if item["filename"] == filename
        ]

        if len(filename_matches) == 1:

            return filename_matches[0][
                "original"
            ]

        if len(filename_matches) > 1:

            raise RuntimeError(
                "\nNo se encontró la ruta exacta "
                "y existen varias imágenes "
                "con el mismo nombre.\n\n"
                f"Referencia CSV:\n{reference}\n\n"
                f"Nombre:\n{filename}\n\n"
                f"Coincidencias: "
                f"{len(filename_matches)}"
            )

        return None

    def copy_member(
        self,
        member: str,
        destination_without_extension: Path,
    ):
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
            self.archive.open(
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

        return destination

    def close(self):
        self.archive.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "GENERADOR DE REFERENCIAS LSM"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Verificar archivos
    # --------------------------------------------------------

    if not LSM_CSV.is_file():

        raise FileNotFoundError(
            f"No existe:\n{LSM_CSV}"
        )

    if not LSM_SCALER.is_file():

        raise FileNotFoundError(
            f"No existe:\n{LSM_SCALER}"
        )

    # --------------------------------------------------------
    # Leer CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        LSM_CSV,
        low_memory=False,
    )

    print()
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

    path_column = find_column(
        df,
        PATH_CANDIDATES,
    )

    split_column = find_column(
        df,
        SPLIT_CANDIDATES,
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
    # Normalizar clase
    # --------------------------------------------------------

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Sólo TRAIN
    # --------------------------------------------------------

    split_values = (
        df[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_mask = (
        split_values == "train"
    )

    df = (
        df.loc[
            train_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"Filas TRAIN: "
        f"{len(df):,}"
    )

    if len(df) != 42743:

        print(
            "[WARN] El número de filas TRAIN "
            "no coincide con 42,743."
        )

    # --------------------------------------------------------
    # Características world_norm
    # --------------------------------------------------------

    X = get_world_norm_matrix(
        df
    )

    print(
        f"Representación: "
        f"{REPRESENTATION}"
    )

    print(
        f"Dimensiones: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Validar NaN / Inf
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
            (~valid_mask).sum()
        )

        print(
            f"[WARN] "
            f"{removed} filas con NaN/Inf."
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
    # Aplicar scaler final LSM
    # --------------------------------------------------------

    scaler = joblib.load(
        LSM_SCALER
    )

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Crear carpeta de salida
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Indexar ZIPs
    # --------------------------------------------------------

    print()
    print(
        "[INFO] Indexando dataset original LSM..."
    )

    lsm_index = ZipImageIndex(
        LSM_ZIP
    )

    print(
        "[INFO] Indexando dataset ASL "
        "para E/L/Z provisionales..."
    )

    asl_index = ZipImageIndex(
        ASL_ZIP
    )

    # --------------------------------------------------------
    # Clases
    # --------------------------------------------------------

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

    manifest = []

    # ========================================================
    # PROCESAR CLASE POR CLASE
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
        # Distancias
        # ----------------------------------------------------

        distances = np.linalg.norm(
            class_features
            - centroid,
            axis=1,
        )

        # ----------------------------------------------------
        # Ordenar muestras por cercanía al centroide
        # ----------------------------------------------------
        #
        # No usamos sólo la primera.
        #
        # Si la más cercana no puede resolverse
        # correctamente en el ZIP, intenta la siguiente.
        # ----------------------------------------------------

        ordered_local_indices = (
            np.argsort(
                distances
            )
        )

        selected = None

        for local_index in ordered_local_indices:

            global_index = int(
                indices[
                    local_index
                ]
            )

            row = df.iloc[
                global_index
            ]

            reference = str(
                row[
                    path_column
                ]
            )

            distance = float(
                distances[
                    local_index
                ]
            )

            # ------------------------------------------------
            # E/L/Z -> ASL
            # ------------------------------------------------

            if class_name in PROVISIONAL_CLASSES:

                member = (
                    asl_index
                    .find_by_reference(
                        reference
                    )
                )

                source_index = (
                    asl_index
                )

                source_dataset = (
                    "ASL_PROVISIONAL"
                )

            # ------------------------------------------------
            # Resto -> LSM
            # ------------------------------------------------

            else:

                member = (
                    lsm_index
                    .find_by_reference(
                        reference
                    )
                )

                source_index = (
                    lsm_index
                )

                source_dataset = (
                    "LSM"
                )

            if member is None:
                continue

            selected = {
                "global_index":
                    global_index,

                "reference":
                    reference,

                "distance":
                    distance,

                "member":
                    member,

                "source_index":
                    source_index,

                "source_dataset":
                    source_dataset,
            }

            break

        # ----------------------------------------------------
        # No se pudo encontrar ninguna imagen
        # ----------------------------------------------------

        if selected is None:

            print(
                f"[ERROR] {class_name}: "
                "no se encontró ninguna imagen "
                "válida."
            )

            manifest.append(
                {
                    "letter":
                        class_name,
                    "status":
                        "IMAGE_NOT_FOUND",
                }
            )

            continue

        # ----------------------------------------------------
        # Copiar imagen
        # ----------------------------------------------------

        destination = (
            selected[
                "source_index"
            ]
            .copy_member(
                selected[
                    "member"
                ],
                OUTPUT_DIR
                / class_name,
            )
        )

        print(
            f"[OK] "
            f"{class_name}: "
            f"{destination.name} "
            f"(dist="
            f"{selected['distance']:.4f}) "
            f"[{selected['source_dataset']}]"
        )

        # ----------------------------------------------------
        # Trazabilidad
        # ----------------------------------------------------

        manifest.append(
            {
                "language":
                    "LSM",

                "letter":
                    class_name,

                "representation":
                    REPRESENTATION,

                "distance_to_centroid":
                    selected[
                        "distance"
                    ],

                "dataset_reference":
                    selected[
                        "reference"
                    ],

                "zip_member":
                    selected[
                        "member"
                    ],

                "source_dataset":
                    selected[
                        "source_dataset"
                    ],

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

    # --------------------------------------------------------
    # Cerrar ZIPs
    # --------------------------------------------------------

    lsm_index.close()
    asl_index.close()

    # ========================================================
    # GUARDAR MANIFIESTO
    # ========================================================

    manifest_path = (
        OUTPUT_DIR
        / "reference_manifest.csv"
    )

    pd.DataFrame(
        manifest
    ).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    successful = [
        item
        for item in manifest
        if item.get(
            "status"
        ) == "OK"
    ]

    print()
    print(
        "=" * 72
    )

    print(
        f"Referencias correctas: "
        f"{len(successful)}/"
        f"{len(classes)}"
    )

    print(
        f"Manifiesto:\n"
        f"{manifest_path}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()