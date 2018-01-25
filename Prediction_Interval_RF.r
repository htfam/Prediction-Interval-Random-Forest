###########################################################################
#                                                                         #
#           Prediction Interval for Random Forest (ranger package)        #
#                                                                         #
#                                                                         #
# Last update: 9-27-17                                                    #
###########################################################################


library(doParallel)

ISU.RFpi = function(rf, train.dat, train.y, test.dat = train.dat, alpha = 0.85) {
  clusters <- makeCluster(detectCores())
  registerDoParallel(clusters)
  if (is.null(rf$inbag.counts)) {
    stop("Random forest must be trained with keep.inbag = TRUE")
  }
  
  pd <- predict(rf, train.dat, type = 'terminalNodes', predict.all = TRUE, num.threads = detectCores())
  chnodes <- pd[["predictions"]]
  pred <- predict(rf, test.dat, type = 'terminalNodes', predict.all = TRUE, num.threads = detectCores())
  pred.nodes <- pred[["predictions"]]
  
  inbag.matrix <- do.call(cbind, rf$inbag.counts)
  
  storage1 <- foreach(i = 1:rf$num.samples, .combine = 'c', .inorder = F) %dopar% {
    leaf.i<-chnodes[i,]
    weight.i<-inbag.matrix
    weight.i[chnodes != matrix(leaf.i,nrow(chnodes),ncol(chnodes),byrow=T)] <-0
    weight.i<-weight.i/matrix(colSums(weight.i),nrow(chnodes),ncol(chnodes),byrow=T)
    avg.weight.i <- rowMeans(weight.i[, which(inbag.matrix[i, ] == 0)]) # out of bag weights for case i
    d <- abs((train.y[i] - rf$predictions[i]) / sqrt(1 + crossprod(avg.weight.i, avg.weight.i)))
  }
  d.a <- quantile(storage1, alpha, na.rm = T) # alpha * 100% sample quantile from d1...dn
  

  storage2 <- foreach(i = 1:nrow(test.dat),.combine = 'rbind') %dopar% {
    leaf.i <- pred.nodes[i,]
    weight.i<-inbag.matrix
    weight.i[chnodes != matrix(leaf.i,nrow(chnodes),ncol(chnodes),byrow=T)] <-0
    weight.i<-weight.i/matrix(colSums(weight.i),nrow(chnodes),ncol(chnodes),byrow=T)
    pred.weight <- rowMeans(weight.i) # prediction weights for test case i
  }

  predicted <- predict(rf, test.dat, 
                       num.threads = detectCores())      
  
  ci.lwr <- predicted$predictions - d.a * sqrt(diag(tcrossprod(storage2)) + 1)
  ci.upr <- predicted$predictions + d.a * sqrt(diag(tcrossprod(storage2)) + 1)
  
  out <- data.frame(pred = predicted$predictions, 
                    ci.lwr = ci.lwr,
                    ci.upr = ci.upr)
  
  stopCluster(clusters)
  registerDoSEQ()
  return(out)
}

