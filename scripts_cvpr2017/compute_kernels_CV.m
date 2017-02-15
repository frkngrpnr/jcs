function [mycel_trainfolds]=compute_kernels_CV(mycel_trainfolds,all_data_train,opt)

Nfolds=numel(mycel_trainfolds);
for i=1:Nfolds
   
    temp_trdata=all_data_train(mycel_trainfolds{i,1}.trainind,:); 
    temp_valdata=all_data_train(mycel_trainfolds{i,1}.testind,:); 
    %clear all_data_train
     if (opt.norm_type==1)
        
         [temp_trdata,setting]=mapminmax(temp_trdata');
         [temp_valdata]=mapminmax('apply',temp_valdata',setting);
         temp_trdata=temp_trdata';
         temp_valdata=temp_valdata';

     elseif (opt.norm_type==2)
   
         [temp_trdata,mx,stdx]=autosc(temp_trdata);
         [temp_valdata]=scal(temp_valdata,mx,stdx);
       
     end
          
  
    if (opt.do_power_norm)
        p=2;
        temp_trdata=sign(temp_trdata).*abs(temp_trdata).^(1/p);
        temp_valdata=sign(temp_valdata).*abs(temp_valdata).^(1/p);
        
    end
    if (opt.do_logsig)
        temp_trdata=logsig(temp_trdata);
        temp_valdata=logsig(temp_valdata);
    end
    
     
    if opt.do_imp_pp
        
        for k=1:size(temp_trdata,1)
            % instance (feature_vector) level L2 norm 
            temp_trdata(k,:)=temp_trdata(k,:)/norm(temp_trdata(k,:));
        end

        for k=1:size(temp_valdata,1)
            temp_valdata(k,:)=temp_valdata(k,:)/norm(temp_valdata(k,:));
        end
    
    end
    if opt.kernel_type==1
        mycel_trainfolds{i,1}.train_kernel=(temp_trdata*temp_trdata');
        mycel_trainfolds{i,1}.val_kernel=(temp_valdata*temp_trdata');
    else
        mycel_trainfolds{i,1}.train_kernel=exp(-opt.gamma*dist(temp_trdata,temp_trdata'));
        mycel_trainfolds{i,1}.val_kernel=exp(-opt.gamma*dist(temp_valdata,temp_trdata'));
    end
     
 
 
    nancnt=sum(sum(isnan(mycel_trainfolds{i,1}.train_kernel)));
    if nancnt>0
        disp([num2str(i) ' NaN:' num2str(nancnt)])
    end
  
    infcnt=sum(sum(isinf(mycel_trainfolds{i,1}.train_kernel)));
     if infcnt>0
        disp([num2str(i) ' Inf:' num2str(infcnt)])
     end 

end
disp('kernels computed');
end