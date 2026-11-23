# Wi‑Fi Vision: CSI to 3D Pose

مشروع تجريبي خفيف يبدأ من بيانات CSI عامة وينتهي بتقدير هيكل إنسان ثلاثي الأبعاد. المسار الحالي يستخدم عينة صغيرة من **Wiception3D / Person-in-WiFi 3D** ولا ينزّل الأرشيف الكامل (نحو 18.7 GB).

## أسرع تشغيل: Google Colab

1. افتح `notebooks/wifi_vision_3d_baseline_colab.ipynb` في Colab.
2. اختر **Runtime → Run all**.
3. وافق مرة واحدة على ربط Google Drive.

يحفظ الدفتر البيانات والنتائج في `MyDrive/WiFi-CSI-Research`. التنزيل قابل للاستئناف، والملفات المكتملة والصحيحة لا تُنزّل مرة أخرى.

## ما نُفّذ

- تحميل 250 زوجًا صغيرًا من CSI/3D labels مباشرة من Kaggle إلى البيئة السحابية.
- التحقق من CSI مركب بأبعاد `(20, 30, 3, 3)` ووسوم شخص واحد `(14, 3)`.
- استخراج ميزات amplitude/phase وتدريب PCA + Ridge محدود الذاكرة.
- تقييم MPJPE مطلق وroot-relative على اختبار زمني ومجموعة التقاط محجوزة.
- حفظ النموذج والرسوم والتنبؤات الصغيرة في `artifacts/baseline_run`.

هذه تجربة سلامة للخط وليست إعادة إنتاج للنموذج البحثي. لا تتوفر هوية الأشخاص في قوائم الملفات العامة المستعملة، لذلك لا نصف التقسيم الحالي بأنه cross-subject.

## تشغيل محلي/سحابي بديل

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_wiception_sample.py --root data/wiception3d_small
.venv/bin/python scripts/run_baseline.py
.venv/bin/python -m pytest -q
```

على Windows استخدم `.venv\Scripts\python.exe` بدل `.venv/bin/python`. لا حاجة لتشغيل ذلك على جهازك إذا استخدمت Colab.

## مصادر البيانات المهمة

- [Person-in-WiFi 3D](https://aiotgroup.github.io/Person-in-WiFi-3D/) — 3D، المصدر الحالي للعينة.
- [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) — 3D ومتعدد الوسائط مع تقسيم cross-subject رسمي؛ المرشح للتوسع لاحقًا.
- [Wi-Pose Dataset](https://github.com/NjtechCVLab/Wi-PoseDataset) — وسوم 2D فقط، وهو مختلف عن نظام WiPose البحثي في MobiCom 2020.

راجع [تقرير الحالة والبيانات](reports/status_ar.md) للتفاصيل والقيود.
