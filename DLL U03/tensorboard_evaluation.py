import tensorflow as tf

class Evaluation:

    def __init__(self, store_dir, session):
        tf.reset_default_graph()
        self.sess = session #tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir, self.sess.graph)

        self.tf_loss = tf.placeholder(tf.float32, name="train_loss_summary")
        tf.summary.scalar("train loss", self.tf_loss)

        # TODO: define more metrics you want to plot during training (e.g. training/validation accuracy)
        self.val_loss = tf.placeholder(tf.float32, name="validation_loss_summary")
        tf.summary.scalar("validation loss", self.val_loss)

        self.tf_acc = tf.placeholder(tf.float32, name="train_accuracy_summary")
        tf.summary.scalar("train accuracy", self.tf_acc)
        self.val_acc = tf.placeholder(tf.float32, name="validation_accuracy_summary")
        tf.summary.scalar("validation accuracy", self.val_acc)
             
        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):
        
       # TODO: add more metrics to the summary 
       summary = self.sess.run(self.performance_summaries,
                               feed_dict={self.tf_loss : eval_dict["loss"],
                                          self.tf_acc  : eval_dict['acc'],
                                          self.val_loss : eval_dict['vloss'],
                                          self.val_acc  : eval_dict['vacc']
                                          })

       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()
