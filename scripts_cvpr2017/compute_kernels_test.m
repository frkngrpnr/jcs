function [mycel_testfold]=compute_kernels_test(mycel_testfold,trdata,valdata,opt)

if (opt.norm_type==1)

     [trdata,setting]=mapminmax(trdata');
     [valdata]=mapminmax('apply',valdata',setting);
     trdata=trdata';
     valdata=valdata';

elseif (opt.norm_type==2)

     [trdata,mx,stdx]=autosc(trdata);
     [valdata]=scal(valdata,mx,stdx);
     %[temp_valdata]=autosc(temp_valdata(:,std_filter),mx,stdx);

end

mycel_testfold{1,1}.gamma=opt.gamma;

if (opt.do_power_norm)
    p=2;
    trdata=sign(trdata).*abs(trdata).^(1/p);
    valdata=sign(valdata).*abs(valdata).^(1/p);

end
if (opt.do_logsig)
    trdata=logsig(trdata);
    valdata=logsig(valdata);
end

if opt.do_imp_pp

    for k=1:size(trdata,1)
        % instance (feature_vector) level L2 norm 
        trdata(k,:)=trdata(k,:)/norm(trdata(k,:));
    end

    for k=1:size(valdata,1)
        valdata(k,:)=valdata(k,:)/norm(valdata(k,:));
    end
end
if opt.kernel_type==1
    mycel_testfold{1,1}.train_kernel=(trdata*trdata');
    mycel_testfold{1,1}.val_kernel=(valdata*trdata');
else
    mycel_testfold{1,1}.train_kernel=exp(-gamma*dist(trdata,trdata'));
    mycel_testfold{1,1}.val_kernel=exp(-gamma*dist(valdata,trdata'));
end

nancnt=sum(sum(isnan(mycel_testfold{1,1}.train_kernel)));
if nancnt>0
    disp([' NaN:' num2str(nancnt)])
end

infcnt=sum(sum(isinf(mycel_testfold{1,1}.train_kernel)));
if infcnt>0
    disp([' Inf:' num2str(infcnt)])
end 

disp('kernels computed');

end