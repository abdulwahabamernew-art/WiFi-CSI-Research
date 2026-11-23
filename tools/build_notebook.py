from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/wifi_vision_3d_baseline_colab.ipynb"


def without_main(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = '\nif __name__ == "__main__":'
    return text.split(marker, 1)[0]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3"}
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Wi‑Fi Vision — baseline ثلاثي الأبعاد من CSI\n\n"
            "هذا الدفتر ينزّل **عينة صغيرة فقط** من Wiception3D المرتبطة رسميًا بمشروع "
            "Person‑in‑WiFi 3D، يفحص CSI والوسوم `14×3`، يدرب baseline خفيفًا، ثم يقيس MPJPE.\n\n"
            "النتيجة هنا اختبار هندسي صغير وليست إعادة إنتاج لنتيجة الورقة. "
            "مجموعة Wi‑Pose المنفصلة ثنائية الأبعاد ولا تُستخدم هنا لإثبات 3D."
        ),
        nbf.v4.new_markdown_cell(
            "## 1. الإعداد والتخزين المستمر\n\n"
            "في Colab سيظهر طلب واحد لربط Google Drive. البيانات والنتائج تحفظ تحت "
            "`MyDrive/WiFiVision` كي لا تضيع بانتهاء الجلسة. خارج Colab يستخدم الدفتر مجلد المشروع الحالي."
        ),
        nbf.v4.new_code_cell(
            "import importlib.util, subprocess, sys, os\n"
            "from pathlib import Path\n\n"
            "required = {'requests':'requests>=2.31', 'h5py':'h5py>=3.10', "
            "'sklearn':'scikit-learn>=1.4', 'joblib':'joblib>=1.3', 'matplotlib':'matplotlib>=3.8'}\n"
            "missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]\n"
            "if missing:\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])\n\n"
            "try:\n"
            "    IN_COLAB = importlib.util.find_spec('google.colab') is not None\n"
            "except ModuleNotFoundError:\n"
            "    IN_COLAB = False\n"
            "if IN_COLAB:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    RUN_ROOT = Path('/content/drive/MyDrive/WiFiVision')\n"
            "else:\n"
            "    RUN_ROOT = Path.cwd()\n"
            "RUN_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "DATA_ROOT = RUN_ROOT / 'data/wiception3d_small'\n"
            "OUTPUT_ROOT = RUN_ROOT / 'artifacts/notebook_run'\n"
            "print({'in_colab': IN_COLAB, 'run_root': str(RUN_ROOT)})"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. تنزيل محدود وقابل للاستئناف\n\n"
            "نستخدم ملفات منفردة من المصدر بدل تنزيل الأرشيف الكامل (نحو 18.7 GB). "
            "الملف المكتمل والصالح لا يُنزّل ثانية، والجزئي يُستأنف عندما يدعم الخادم طلبات النطاق."
        ),
        nbf.v4.new_code_cell(without_main(ROOT / "scripts/download_wiception_sample.py")),
        nbf.v4.new_code_cell(
            "manifest_path = build_sample(DATA_ROOT, samples_per_group=50, workers=4)\n"
            "manifest_path"
        ),
        nbf.v4.new_markdown_cell("## 3. فحص CSI والوسوم ثلاثية الأبعاد"),
        nbf.v4.new_code_cell((ROOT / "src/wifi_vision.py").read_text(encoding="utf-8")),
        nbf.v4.new_code_cell(
            "import json\n"
            "manifest = json.loads(manifest_path.read_text())['samples']\n"
            "row = manifest[0]\n"
            "sample_csi = load_complex_csi(row['csi_path'])\n"
            "sample_pose = load_single_person_pose(row['pose_path'])\n"
            "print('CSI shape:', sample_csi.shape, sample_csi.dtype)\n"
            "print('Pose shape:', sample_pose.shape, sample_pose.dtype)\n"
            "print('Feature length:', csi_features(sample_csi).shape[0])"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. تدريب baseline وتقييمه\n\n"
            "التدريب على مجموعتي الالتقاط S11 وS21. نقيس اختبارًا زمنيًا من المجموعتين نفسيهما، "
            "ثم اختبار نقل أصعب على S51 المحجوزة بالكامل. المقياس MPJPE بالملليمتر."
        ),
        nbf.v4.new_code_cell(
            "metrics = evaluate_and_save(manifest_path, OUTPUT_ROOT, max_components=64, alpha=20.0)\n"
            "print(json.dumps(metrics, indent=2, ensure_ascii=False))"
        ),
        nbf.v4.new_code_cell(
            "pred = np.load(OUTPUT_ROOT / 'test_predictions.npz')\n"
            "heldout_row = next(r for r in manifest if r['split']=='test' and r['capture_group']=='S51')\n"
            "plot_csi_and_pose(\n"
            "    load_complex_csi(heldout_row['csi_path']),\n"
            "    load_single_person_pose(heldout_row['pose_path']),\n"
            "    pred['heldout_prediction'][0],\n"
            "    OUTPUT_ROOT / 'csi_and_pose.png',\n"
            ")\n"
            "plot_joint_errors(metrics, OUTPUT_ROOT / 'joint_errors.png')\n"
            "from IPython.display import display, Image\n"
            "display(Image(filename=str(OUTPUT_ROOT / 'csi_and_pose.png')))\n"
            "display(Image(filename=str(OUTPUT_ROOT / 'joint_errors.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. ما الذي تثبته التجربة وما الذي لا تثبته؟\n\n"
            "- تثبت أن خط البيانات يعمل فعليًا من CSI مركب إلى إحداثيات مفاصل 3D ومقياس MPJPE.\n"
            "- لا تثبت جودة عملية بعد؛ العينة صغيرة وRidge أبسط كثيرًا من Transformer الأصلي.\n"
            "- هوية المتطوع غير موجودة في قوائم الملفات العامة المستخدمة هنا، لذلك هذا ليس تقسيمًا حسب الأشخاص.\n"
            "- الفرق الكبير عند حجز S51 يوضح حساسية CSI لتغير مجموعة الالتقاط/البيئة.\n"
            "- الخطوة التالية الجادة هي MM‑Fi بتقسيم `cross_subject_split` أو Wiception3D كاملة مع metadata للهوية."
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    build()

