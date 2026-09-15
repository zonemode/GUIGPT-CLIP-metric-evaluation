# GUIGPT-CLIP-metric-evaluation
1.选用 ViT-B/32、FG-CLIP2-Base、EVA02-CLIP-B-16 三种CLIP模型，对 Figma、LayoutCoder、ScreenShotToCode、LayoutCoder_FullPAE、GUIGPT 五种算法生成的UI图片进行 CLIP-I 指标计算，GUIGPT 均获最优，在三种模型下较前四种算法平均提升约 8.5%、16.7%和20.8%。
2.基于 PaddleOCR 开发批量自动化文字提取流程，精准识别五种算法生成图中的文字内容与空间位置信息，输出结构化JSON文件及带标注框的可视化图片，支撑后续人工校验与定量分析。
3.引入 HiMo-CLIP 模型进行 CLIP Score 语义对齐评估，GUIGPT 仍为最优，较前四种算法平均提升约 9.1%，并探索了 FG-CLIP2-Base 与 HiMo-CLIP 的复合模型融合策略。
