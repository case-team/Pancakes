cmsDriver.py pancakes_2017_mc -n 100 --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --conditions 102X_mc2017_realistic_v7 --step NANO --nThreads 4 \
    --era Run2_2017,run2_nanoAOD_94XMiniAODv2 --customise PhysicsTools/Pancakes/nanoHRT_cff.nanoHRT_customizeMC \
    --customise PhysicsTools/Pancakes/ak8_cff.addCustomizedAK8PF \
    --filein file:miniAOD.root --fileout file:nano_mc_2017.root \
    --no_exec --customise Configuration/DataProcessing/Utils.addMonitoring

python crab.py -p pancakes_2017_mc_NANO.py -o /store/user/oamram/case/2017/ -t pancakes-02 -i CASE_signals_2017.txt \
    --num-cores 4 --max-memory 6000 --send-external -s EventAwareLumiBased -n 50000 --work-area crab_projects_mc_2017 
