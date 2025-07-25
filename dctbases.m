function D = dctbases(N,K)
    D = zeros(N,K);
    for k = 1:K
        d = cos(pi*(0:N-1)*(k-1)/K)';
        if k>1
            d = d-mean(d);
        else
            d = d-0;
        end
        D(:,k) = d/norm(d);
    end
end