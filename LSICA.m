function [T,S,Err,C]= LSICA(Y,K,spa,nIter,TC,SM)
 
% Prewhitening with pseudoinverse
[F, G, ~] = svds(Y, K);
Xs = pinv(G) * F' * Y;
U = ones(K,K);
S = U'*Xs;

fprintf('Iteration:     ');
for j= 1:nIter
    Uo =U;
    fprintf('\b\b\b\b\b%5i',j);
 
    tmp = U'*Xs;
    S = sign(tmp).*max(0, bsxfun(@minus,abs(tmp),spa/2));
    [F1, ~, G1] = svds(Xs*S',K);
    U = F1*G1';

    Err(j) = (sqrt(trace((U-Uo)'*(U-Uo)))/sqrt(trace(Uo'*Uo)));     


    T = Y*S';
        
    K2 = size(TC,2);
    [~,~,ind]=sort_TSandSM_spatial(TC,SM,T,S,K2);
    for ii =1:K2
        TCcorr(ii) =abs(corr(TC(:,ii),T(:,ind(ii))));
        SMcorr(ii) =abs(corr(SM(ii,:)',S(ind(ii),:)'));
    end
    cTC = sum(TCcorr');
    cSM = sum(SMcorr');
    C(j) =cTC+cSM;

end


