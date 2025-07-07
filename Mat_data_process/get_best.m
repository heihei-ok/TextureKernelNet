
% Calculate the optimal parameters

L_ALL = [150000,150000,150000,800000,1200000];
lay_num = 10:1:50;
win = 224;
min_lay_num = 150000 / (224*224); % The minimum window quantity is determined under the condition of the maximum scanning speed.
mmax_lay_num = 1200000/ (224*224);


x1=cell(1,size(lay_num,2));
sita = [1,1,1,fix(80/15),8];
for num =  10:1:50
    L = L_ALL / num;
    sigma_max = floor((L(1) - 224) / 223);
    %  When setting TAO to N, it exactly equals L(1), and no point will be wasted.
    x = [];
    for sigm = 1:sigma_max
        loss = 0;
        sigm_2 = sigm * sita;
        L_2 = ((win-1) * sigm_2) + 224;
        loss = mean((L_2)./L);
        x = [x,loss];
    end
    x1{1,num-9}=x;
end

%% 

f = size(x1,2);
max = size(x1{1,1}, 2);
chendu = zeros(f,max);
count = 0;
count_x = 0;
count_y = 0;
max_n = zeros(1,f);
for i = 1:f
    m = x1{1,i};
    x = repmat(i,1,max);
    y = 1:max;
    for j = 1:size(m, 2)

    chendu(i,j) = m(1,j);
    max_n(1,i) = m(1,size(m,2));
       if m(1,j) >count
           count = m(1,j);
           count_x = i;
           count_y = j;
       end
    end
    
end


%% 
% 2D
imagesc(chendu)
% 3D
surf(chendu)











