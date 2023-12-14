# coding=gb2312
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import datasets
from sklearn import svm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak,Image,Table
from reportlab.graphics.shapes import Image as DrawingImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors


# DATAPATH = 'C:Users/Administrator/Desktop/临时/LR_demo/DATAPATH/'
# " DATAPATH 和 filePTH  根据自己的进行修改"
# filePath = 'C:Users/Administrator/Desktop/临时/LR_demo/pv_alpha/'

DATAPATH = 'B:\PV_code_local\PV_code\LR_code\DATAPATH'
" DATAPATH 和 filePTH  根据自己的进行修改"
# filePath = '/home/intern/PV_code/PV_code/LR_code/pv_alpha'

#设置字体
h3 = ParagraphStyle(name="h3", fontSize=16, leading=21, alignment=1)
h4 = ParagraphStyle(name="h4", fontSize=12, leading=21, alignment=1)
h5 = ParagraphStyle(name="h4", fontSize=13, leading=21, alignment=0)
from load_dataset import *

if __name__ == "__main__":

    start_num = 0  # add by brian
    end_num = 2  # add by brian

    # create pdf, add by kai
    current_directory = os.getcwd()
    modelType = "Linear Regression"
    output_filename = "Output/" + modelType + " analyze.pdf"
    data_path = os.path.join(current_directory, output_filename)
    # 调用模板，创建指定名称的PDF文档
    doc = SimpleDocTemplate(data_path)
    # 获得模板表格
    styles = getSampleStyleSheet()
    # 指定模板
    style = styles['Normal']
    # 初始化内容
    TextInter = []
    TextInter.append(Paragraph(f"Analyze<br></br><br></br>", h3))
    TextInter.append(Paragraph("Parameters : <br></br>", h5))
    TextInter.append(Paragraph(f"ModelType : {modelType}.<br></br>", style))
    TextInter.append(Paragraph(f"Alpha factors : {end_num} ", style))

    TextInter.append(Paragraph("<br></br><br></br>Results : <br></br>", h5))

    output_list = []
    for year_num in range(2010, 2012, 1):
        print('当前预测年份为：', year_num)


        X_train, Y_train, X_test, Y_test, X_predict, Y_predict = GeneNewdata(start_num, end_num, current_date=str(year_num),
                                                                             sample_windows_in_month=24,
                                                                             sample_windows_out_windows=12)
        # 获取数据
        alpha_train, train_sample = X_train.shape
        N_train_y = Y_train.shape
        alpha_test, test_sample = X_test.shape
        N_test_y = Y_test.shape

        # 创建线性回归模型
        model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1)
        # 在训练集上拟合模型
        model.fit(X_train.T, Y_train)
        # 在内部数据集上检测模型的性能  add by brian
        Y_pred_inter = model.predict(X_train.T)
        mse_inter = mean_squared_error(Y_train, Y_pred_inter)
        RMSE_inter = np.sqrt(mse_inter)
        MAE_inter = mean_absolute_error(Y_train, Y_pred_inter)
        r2_inter = r2_score(Y_train, Y_pred_inter)
        in_corr_ = np.corrcoef(Y_pred_inter, Y_train.values)

        # 交叉验证 add by kai
        # clf = svm.SVC(kernel='linear', C=1).fit(X_train.T, Y_train)
        # clf_scores = cross_val_score(clf, X_test.T, Y_test, cv=5)
        mse_scores_inter = cross_val_score(model, X_train.T, Y_train, scoring='neg_mean_squared_error', cv=3)
        mse_scores_inter = -mse_scores_inter

        # 建立一个pdf文档，用于将结果输出到文档上   add by brian
        # current_directory = os.getcwd()
        # output_filename = "Output/" + f"{start_num - end_num + 1}alpha.pdf"
        # data_path = os.path.join(current_directory,output_filename)

        # add by kai


        # 将段落添加到内容中
        # strPara1 = "This is the analyze of " + str(end_num) + " alpha factors"
        # strPara1 = f"This is the analyze of {end_num} alpha factors.<br></br>"



        TextInter.append(Paragraph(f'<br></br>Current year of predict : {year_num}',h4 ))
        TextInter.append(Paragraph("<br></br>Inter Data Evaluation:",style))
        TextInter.append(Paragraph(f"Mean Squared Error (MSE): {mse_inter}", style))
        TextInter.append(Paragraph(f"Root Mean Squared Error (RMSE): {RMSE_inter}", style))
        TextInter.append(Paragraph(f"Mean Absolute Error (MAE): {MAE_inter}", style))
        TextInter.append(Paragraph(f"R-squared (R2): {r2_inter}", style))
        TextInter.append(Paragraph(f"Corr is : {in_corr_}", style))
        TextInter.append(Paragraph(f"cross-validation is : {mse_scores_inter}", style))




        # 输出训练集上评估结果  add by brian
        print("Inter Data Evaluation:")
        print("Mean Squared Error (MSE):", mse_inter)
        print("R-squared (R2):", r2_inter)
        print("Corr is :", in_corr_)
        print("cross-validation is :", mse_scores_inter)


        # 在外部数据集上进行预测
        Y_pred_outer = model.predict(X_test.T)
        # 评估模型在预测数据集上的性能
        mse_outer = mean_squared_error(Y_test, Y_pred_outer)
        RMSE_outer = np.sqrt(mse_inter)
        MAE_outer = mean_absolute_error(Y_test, Y_pred_outer)
        r2_outer = r2_score(Y_test, Y_pred_outer)
        corr_ = np.corrcoef(Y_pred_outer, Y_test.values)
        mse_scores_outer = cross_val_score(model, X_test.T, Y_test, scoring='neg_mean_squared_error', cv=3)
        mse_scores_outer = -mse_scores_outer


        # add by kai
        # 初始化内容
        # TextOuter = []
        # 将段落添加到内容中
        # TextOuter.append(Paragraph("This is the analyze of " + end_num + " alpha factors", style))
        TextInter.append(Paragraph("<br></br>Outer Data Evaluation:"))
        TextInter.append(Paragraph(f"Mean Squared Error (MSE): {mse_outer}", style))
        TextInter.append(Paragraph(f"Root Mean Squared Error (RMSE): {RMSE_outer}", style))
        TextInter.append(Paragraph(f"Mean Absolute Error (MAE): {MAE_outer}", style))
        TextInter.append(Paragraph(f"R-squared (R2): {r2_outer}", style))
        TextInter.append(Paragraph(f"Corr is : {corr_}", style))
        TextInter.append(Paragraph(f"cross-validation is : {mse_scores_outer}", style))
        # doc.build([TextInter, PageBreak(),Paragraph("这是第二页")])


        # 输出评估结果
        print("Outer Data Evaluation:")
        print("Mean Squared Error (MSE):", mse_outer)
        print("R-squared (R2):", r2_outer)
        print("Corr is :", corr_)
        print("cross-validation is :", mse_scores_outer)
        # Predict Data
        Y_predict_output = model.predict(X_predict.T)
        Y_predict_LR = pd.DataFrame(Y_predict_output, index=Y_predict.index).reset_index().pivot(index='trade_dt',
                                                                                                 columns='s_info_windcode')
        Y_predict_LR.columns = Y_predict_LR.columns.get_level_values(1)
        output_list.append(Y_predict_LR)

    # doc.build(TextInter)
    # factor_path = '/home/wenrr/Code/Jupyter/ProjectPV/factor_file/'
    # factor.to_hdf(os.path.join(factor_path,'PV_linear_LR.h5'),key='w')

    # %% BackTest 回测
    factor = pd.concat(output_list)
    factor.columns = factor.columns.map(lambda x: x[:6])
    from BackTest_v0 import StockBackTest

    BT = StockBackTest(factor)
    BT.grouptest_gen(10)
    # BT.draw_groups(mode='net')

    BT.draw_groups(mode='group')
    picPath = os.path.join(current_directory, "Output/" + "group.jpg")

    #test for open image
    img = plt.imread(picPath)
    fig = plt.figure('show picture')
    plt.imshow(img)
    plt.show()

    TextInter.append(Image(picPath))
    BT.backtest()
    print(BT.summary())

    #将金融数据输入到pdf中
    summary = BT.summary()
    TextInter.append(Paragraph("<br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br>", style))
    TextInter.append(Paragraph("Financial data", h5))
    TextInter.append(Paragraph("<br></br><br></br>", style))
    # dicSummary = summary.to_dict('dict')
    # financialTable = Table(summary)
    # list = [summary.columns[:, ].values.astype(str).tolist()] + summary.values.tolist()
    # print("--------before reset_index summary.index---------")
    # print(summary.index)

    # print("--------after reset_index summary.index---------")
    # print(summary.index)
    summary = summary.reset_index()
    # indexYear = summary.values[:,0:1].astype(str).tolist()
    indexYear = summary.values[:, 0:1].astype(str)
    # print(indexYear)
    # print(indexYear.dtype)
    # print(summary.values[:,1:9].dtype)
    # print(summary.values[:, 0:9].astype(str).dtype)
    # print(summary.values[:,1:9].astype(str).dtype)
    listA = [['year'] + summary.columns[1:8, ].values.astype(str).tolist()] + (summary.values[:,0:8]).astype(str).tolist()
    listB = [['year'] + summary.columns[8:16, ].values.astype(str).tolist()] + np.hstack((summary.values[:, 0:1],summary.values[:,8:16])).astype(str).tolist()
    listC = [['year'] + summary.columns[16:, ].values.astype(str).tolist()] + np.hstack((summary.values[:, 0:1],summary.values[:,16:])).astype(str).tolist()
    # listB = [summary.columns[8:16, ].values.astype(str).tolist()] + summary.values[:, 9:17].astype(str).tolist()
    # listC = [summary.columns[16:, ].values.astype(str).tolist()] + summary.values[:,17: ].astype(str).tolist()

    print("--------summary.columns---------")
    print(indexYear)
    # print(summary.index.tolist())
    print(listA)
    print(listB)
    print(listC)
    # print(summary.values[:,0:8].astype(str).tolist())
    # print(summary.values[:,8:16].astype(str).tolist())
    # print(summary.values[:,16:].astype(str).tolist())
    ts = [('ALIGN', (1, 1), (-1, -1), 'CENTER'),
          ('LINEABOVE', (0, 0), (-1, 0), 1, colors.purple),
          ('LINEBELOW', (0, 0), (-1, 0), 1, colors.purple),
          ('FONT', (0, 0), (-1, 0), 'Times-Bold'),
          ('LINEABOVE', (0, -1), (-1, -1), 1, colors.purple),
          ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.purple, 1, None, None, 4, 1),
          ('LINEBELOW', (0, -1), (-1, -1), 1, colors.red),
          ('FONT', (0, -1), (-1, -1), 'Times-Bold'),
          # ('BACKGROUND', (1, 1), (-2, -2), colors.green),
          # ('TEXTCOLOR', (0, 0), (1, -1), colors.red)
          ]
    tableA = Table(listA, style=ts)
    tableB = Table(listB, style=ts)
    tableC = Table(listC, style=ts)
    TextInter.append(tableA)
    TextInter.append(Paragraph("<br></br><br></br>", style))
    TextInter.append(tableB)
    TextInter.append(Paragraph("<br></br><br></br>", style))
    # TextInter.append(Paragraph(""),style)
    TextInter.append(tableC)
    # TextInter.append(str(summary))
    # TextInter.append(Paragraph(f"ret : {summary.ret}", style))
    # TextInter.append(Paragraph(f"ret : {summary.ret}", style))

    # BT.draw_picture(BT.start_date, BT.end_date);

    # 将内容输出到PDF中
    doc.build(TextInter)