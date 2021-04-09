rm -rf pytorch_structure2vec 
git clone https://github.com/Hanjun-Dai/pytorch_structure2vec.git
cd pytorch_structure2vec
git checkout 848db859f471ad010c1465534251d0bf6ac5531c
cd ..
mkdir dropbox
curl -L -o dropbox/data.zip https://www.dropbox.com/sh/mu8odkd36x54rl3/AABg8ABiMqwcMEM5qKIY97nla?dl=1
unzip dropbox/data.zip -d dropbox
rm dropbox/data.zip
