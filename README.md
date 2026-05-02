# MT-domain-adaptation


General parallel corpus CA-EU
https://huggingface.co/datasets/projecte-aina/CA-EU_Parallel_Corpus


EU

already downloaded 

https://www.ehu.eus/ehg/kc/  

https://github.com/hltfbk/E3C-Corpus



CA  


https://ctilc.iec.cat/scripts/CTILCCorpus_Descarr.asp


TO DO:
- Prepare three fine-tuning scripts: general, literary, and clinical
- Keep the general model as a standalone baseline
- Fine-tune the in-domain models (literary and clinical separate) starting from the general model
- Run experiments fine-tuning the in-domain data directly from the base model (without general-domain fine-tuning) for comparison
- Evaluate Latxa baseline alone, the general-domain model, the general→literary and general→clinical models, as well as the literary-only and clinical-only models



ca-literary: translate es to eu (pivoting)  DONE

eu-literary: translate es to ca (pivotong)  IN PROGRESS

eu-clinical: translate eu to ca directly IN PROGRESS




