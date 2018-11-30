import tensorflow as tf

class Evaluation:

    def __init__(self, store_dir, session):
        tf.reset_default_graph()

        # <JAB>
        self.sess = session #tf.Session()


        self.tf_writer = tf.summary.FileWriter(store_dir, self.sess.graph)

        with self.sess.graph.as_default():
            self.tf_loss = tf.placeholder(tf.float32, name="loss_summary")
            tf.summary.scalar("loss", self.tf_loss)

            self.val_loss = tf.placeholder(tf.float32, name="validation_loss_summary")
            tf.summary.scalar("validation loss", self.val_loss)

            self.tf_acc = tf.placeholder(tf.float32, name="train_accuracy_summary")
            tf.summary.scalar("train accuracy", self.tf_acc)

            self.val_acc = tf.placeholder(tf.float32, name="validation_accuracy_summary")
            tf.summary.scalar("validation accuracy", self.val_acc)

            self.performance_summaries = tf.summary.merge_all()

        # </JAB>

    def write_episode_data(self, episode, eval_dict):
        
       # TODO: add more metrics to the summary
       with self.sess.graph.as_default():
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
