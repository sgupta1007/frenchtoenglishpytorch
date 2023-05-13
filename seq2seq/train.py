import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch_utils import MAX_LENGTH,device,SOS_TOKEN,teacher_forcing_ratio,EOS_TOKEN,hidden_size
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from data_utils import tensorsFromPair,pairs,input_lang,output_lang

def oneepoch(input_tensor,target_tensor,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ec in range(input_length):
        encoder_output,encoder_hidden=encoder(input_tensor[ec],encoder_hidden)
        encoder_outputs[ec]=encoder_output[0,0]
    
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    
    decoder_hidden=encoder_hidden

    use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
             decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
             topv, topi = decoder_output.topk(1)
             decoder_input = topi.squeeze().detach()
             loss += criterion(decoder_output, target_tensor[di])
             if decoder_input.item() == EOS_TOKEN:
                 break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def training(encoder, decoder, n_iters,learning_rate=0.01):
    encoder_optim=optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optim=optim.SGD(decoder.parameters(),lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs))for i in range(n_iters)]
    criterion=nn.NLLLoss()
    losses=[]

    for iter in range(1,n_iters+1):
        training_pair=training_pairs[iter-1]
        input_tensor=training_pair[0]
        target_tensor=training_pair[1]

        loss=oneepoch(input_tensor=input_tensor,target_tensor=target_tensor,criterion=criterion,encoder=encoder,decoder=decoder,encoder_optimizer=encoder_optim,decoder_optimizer=decoder_optim,max_length=MAX_LENGTH)
        losses.append(loss)
    return losses


#training 

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
        
losses=training(encoder1,attn_decoder1,n_iters=75000)

torch.save(encoder1.state_dict(),"encoder1.pt")
torch.save(attn_decoder1.state_dict(),"att_decoder1.pt")


losses=torch.tensor(losses)
torch.save(losses,"losses.pt")

#print("losses",losses)

#plt.xlabel("epochs")
#plt.ylabel("loss")
#plt.title("training losses")
#plt.plot(list(range(len(losses))),losses)
#plt.savefig("training_loss.png")


