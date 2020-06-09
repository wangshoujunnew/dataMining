"""
tensorflow工具
"""
import tensorflow as tf


def convert_tensor(datas, types):
    """
    将测试数据类型转为批量转为想要的数据类型
    """
    return list(
        map(lambda tuple2: tf.constant(tuple2[0], tuple2[1]), zip(datas, types))
    )


def _look_var(vars, session):
    """
    将变量, 在session中打印出shape和其值
    """
    for x in [session.run(var) for var in vars]:
        print("=" * 50)
        if hasattr(x, "shape"):
            print(f"shape: {x.shape}, value:")
        print(x)


def look_var(look_var_arr):
    with tf.Session() as session:
        init_val = tf.compat.v1.global_variables_initializer()
        init_table = tf.compat.v1.tables_initializer()

        session.run([init_table, init_val])

        _look_var(look_var_arr, session)
