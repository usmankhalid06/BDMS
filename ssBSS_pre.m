 function [T,S,Err,CC]= ssBSS_pre(Y,Dp,params,TC,SM)
 
K = params.K;
P = params.P;
lam1 = params.lam1; 
zeta1 = params.zeta1;
Kp = params.Kp;
nIter = params.nIter;

[F, V, D]= svds(Y,K);
Xt = V*D'*Y';
Xs = F'*Y;
Xt = (Xt'*diag(1./sqrt(sum(Xt'.*Xt'))))';
rng(1,'twister'); T = randn(P,size(Y,1));   T = (T'*diag(1./sqrt(sum(T'.*T'))))';  S = pinv(T')*Y;
Dp2 = Dp(:,1:Kp);
U = zeros(K,P);
W = zeros(K,P);
eps = 10^-9;

% fprintf('Iteration:     ');
for j= 1:nIter
%     fprintf('\b\b\b\b\b%5i',j);
    Told = T; 
    
    T = (Y*S'*inv((S*S') + eps*speye(size(S,1))))'; T= T'; T = T*diag(1./sqrt(sum(T.*T))); T= T';  
    A = zeros(size(Dp2,2),P);
    U = Xt*T'*inv((T*T') + eps*speye(size(T,1)));
    T = inv(U'*U+ eps*speye(size(U,2)))*U'*Xt;
    T= T';
    for k = 1:P
        [~,bb]= sort(abs(Dp2'*T(:,k)),'descend');
        ind = bb(1:zeta1);
        A(ind,k)= (Dp2(:,ind)'*Dp2(:,ind))\Dp2(:,ind)'*T(:,k);
        A(:,k) = A(:,k)./norm(Dp2*A(:,k));
        T(:,k) = Dp2*A(:,k);
    end
    T = T';
   
    S = inv((T*T') + eps*speye(size(T,1)))*T*Y;
    W = Xs*S'*inv((S*S') + eps*speye(size(S,1))); 
    tmp2 = inv(W'*W+ eps*speye(size(W,2)))*W'*Xs;
    for i =1:P 
       S(i,:) = sign(tmp2(i,:)).*max(0, bsxfun(@minus,abs(tmp2(i,:)),lam1/2));
       if (length(find(S(i,:)))<1)
           sprintf('Replacing bad source')
           [~,ind]= max(sum(Y-T'*S.^2));
           T(i,:)= (Y(:,ind)./norm(Y(:,ind),2))';
           S(i,:) = 0;
           S(i,:) = sign(T(i,:)*Y).*max(0, bsxfun(@minus,abs(T(i,:)*Y),lam1/2));
       end
    end

    Err(j) = (sqrt(trace((T-Told)'*(T-Told)))/sqrt(trace(Told'*Told)));     
    
    [~,~,ind]=sort_TSandSM_spatial(TC,SM,T',S,P);
    for ii =1:P
        TCcorr(ii) =abs(corr(TC(:,ii),T(ind(ii),:)'));
        SMcorr(ii) =abs(corr(SM(ii,:)',S(ind(ii),:)'));
    end
    cTC = sum(TCcorr');
    cSM = sum(SMcorr');
    CC(j) =cTC+cSM;
    
end

T=T';
Xs= (Xs'*diag(1./sqrt(sum(Xs'.*Xs'))))';        