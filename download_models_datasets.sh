# download datasets
mkdir datasets
cd datasets
wget https://www.dropbox.com/s/xq4m5szng6mkifi/OddOneOutEnv-N4-4C2S2S1-hardMode-UseBGFalse-AgentPos0505-WoAgentFalse-OcclusionFalse-SkewedFalse-Seed0-Tr50000-Val10000.hdf5
wget https://www.dropbox.com/s/xq4m5szng6mkifi/OddOneOutEnv-N4-4C2S2S1-hardMode-UseBGFalse-AgentPos0505-WoAgentFalse-OcclusionFalse-SkewedFalse-Seed0-Tr50000-Val10000.hdf5
wget https://www.dropbox.com/s/j99rp96xw4ruw5l/PushEnv-N3-3C4S1S1-hardMode-UseBGFalse-AgentPos0505-WoAgentFalse-OcclusionFalse-SkewedFalse-Seed0-Tr50000-Val10000.hdf5
wget https://www.dropbox.com/s/o8nmi63gq1lptvn/TargetEnv-N4-4C4S3S1-hardMode-UseBGFalse-AgentPos0505-WoAgentFalse-OcclusionFalse-SkewedFalse-Seed0-Tr50000-Val10000.hdf5
wget https://www.dropbox.com/s/8m0aytl4wkinmqe/RandomObjsEnv-N5C4S4-AgentPosNo-WoAgentTrue-OcclusionTrue-SkewedFalse-Tr1000000-Val10000.hdf5
cd ..
# download trained models
mkdir trained_models
mkdir trained_models/mae-vitb-patch16
mkdir trained_models/mae-vitb-patch16/19ve6sfu
cd trained_models/mae-vitb-patch16/19ve6sfu
wget https://www.dropbox.com/s/l8aju3gm0edk0r2/model_best.pth
