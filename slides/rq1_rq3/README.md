# RQ1 / RQ3 幻灯片

本目录只存放当前同学负责的 RQ1/RQ3 内容，与 `slides/zby/` 的 ZBY 原始材料分开：

- `main.tex`：使用 ZBY Beamer 风格的独立编译入口；
- `frames.tex`：13 个可合并 frame，前 10 个为正式内容，后 3 个为备查；
- `speaker_notes.md`：约 7--8 分钟逐页讲稿。

在项目根目录编译：

```bash
make rq-slides
```

中间文件写入 `/tmp/automl-market-latex/rq-slides`，最终文件为 `deliverables/rq1_rq3_slides.pdf`。

若要并入 ZBY 的 `C3_Paper_Theory.tex`，其导言区图片路径应包含：

```tex
\graphicspath{{./figures/}{../../results/figures/}{../../docs/assets/}}
```

然后在需要的位置加入：

```tex
\input{../rq1_rq3/frames.tex}
```

不要把本目录中的文件移入 `slides/zby/`；该目录只保留 ZBY 的原始理论材料和模板。
