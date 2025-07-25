function [srtd_Zt,srtd_Zs,ind_Zs]=sort_TSandSM_spatial(TC,SM,Zt,Zs,srcs) 
   for j=1:srcs
        [~, ind_Zs(j)]  = max(abs(corr(abs(SM(j,:)'),abs(Zs'))));
        srtd_Zs(j,:) =  Zs(ind_Zs(j),:);
        srtd_Zt(:,j) = sign(corr(TC(:,j),Zt(:,ind_Zs(j))))*Zt(:,ind_Zs(j)); 
    end  