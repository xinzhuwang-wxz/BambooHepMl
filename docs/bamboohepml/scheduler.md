# Scheduler Module

调度器模块。

## 核心组件

### BaseScheduler

调度器基类，定义调度器接口。

::: bamboohepml.scheduler.base.BaseScheduler
    options:
      show_source: true
      heading_level: 3

### LocalScheduler

本地调度器，直接在本地执行任务。

::: bamboohepml.scheduler.local.LocalScheduler
    options:
      show_source: true
      heading_level: 3

### SLURMScheduler

SLURM 调度器，使用 SLURM 提交任务到集群。

::: bamboohepml.scheduler.slurm.SLURMScheduler
    options:
      show_source: true
      heading_level: 3
