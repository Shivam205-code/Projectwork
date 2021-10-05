Import the Data of DST cycle for 45 Degree
data = readmatrix('DST45.csv');
x = data(:,1:2);
y = data(:,6);
m = length(y);

Moving Average Filter
a = 1;
b = ones(1,7200)/7200;
Xf = filter(b,a,x);

Data Normalization
y2 = log(1+y);
for i = 1:2;
    Xn(:,i) = -1 + 2.*(Xf(:,i) - min(Xf(:,i)))./(max(Xf(:,i)) - min(Xf(:,i)));
end
plot(Xn(:,i),y2,'o');

Train Artificial Neural Network Using BPNN
Xt = Xn';
Yt = y2';
hiddenLayerSize = 10;
performancegoal = 0.000001;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;
net = feedforwardnet(10);
[net,tr] = train(net,Xt,Yt);

Performance of ANN
SOCestimateT= (exp(net(Xt(:,tr.trainInd)))-1)*100;
SOCtrueT = (exp(Yt(tr.trainInd))-1)*100;
sqrt(mean((SOCestimateT - SOCtrueT).^2))*100

SOCestimateV = (exp(net(Xt(:,tr.valInd)))-1)*100;
SOCtrueV = (exp(Yt(tr.valInd))-1)*100;
sqrt(mean((SOCestimateV - SOCtrueV).^2))*100

Expected Graph
plot(SOCtrueT)
plot(SOCestimateT)
