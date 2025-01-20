    def loglike_loss_old(self, batch, forward=True):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        seq_obj = Sequence(batch, self.device, self.dtime_transform, self.len_transform, self.interarrival_time_transform)
        # apply embedding on the delay sequence
        embedding = self.encode(seq_obj)
        # embedding: [batch_size, seq_len, d_model]

        # check if we are running validation or training
        if not forward:
            dtime_loss, num_predictions = self.loglike_loss_eval(batch)
            return dtime_loss, num_predictions.item(), None, None
        
        # training: forward pass
        seq_mask = seq_obj.non_pad_mask.float()
        seq_lengths = seq_mask.squeeze(-1).sum(dim=1).long()
        packed_input = pack_padded_sequence(embedding, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.layer_rnn(packed_input)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        # rnn_out: [batch_size, seq_len, d_model(*2 if bidirectional)]
        pred_dist = self.get_pred_distribution(rnn_out)
        # result: [batch_size, seq_len]
        
        # 6) Compute negative log-likelihood vs. the ground-truth times
        assert seq_obj.dtime_seqs_transformed.shape == pred_dist.mean.shape # [batch_size, seq_len]
        dtime_ll = pred_dist.log_prob(seq_obj.dtime_seqs_transformed) * seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()
        dtime_loss = -dtime_ll.sum()

        num_predictions = (seq_obj.non_pad_mask.long() * seq_obj.label_mask_seqs.long()).sum()
        
        return dtime_loss, num_predictions.item(), None, None



                pred_dtime_step = pred_dist_step.mean
                pred_dtime_step_transformed = self.dtime_transform.inv(pred_dtime_step)
                enc_last_pred = torch.zeros((1,1,self.d_model), device=self.device)
                if self.include_dtime_embedding:
                    dtime_enc = self.dtime_emb_layer(pred_dtime_step_transformed.unsqueeze(-1))
                    enc_last_pred += dtime_enc
                if self.include_interarrival_time_embedding:
                    interarrival_time_enc = self.layer_interarrival_time_embedding(interarrival_time_batch.unsqueeze(-1))
                    enc_last_pred += interarrival_time_enc
                input_step = enc_last_pred
